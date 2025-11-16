from src.load_track import *
from config import speed
def simulate_vehicle_step(state, control, track, cfg, dt):
    """
    Full 2D bicycle model + energy + battery + track coordinates.
    """

    # ----------------------------
    # 1. Extract variables
    # ----------------------------
    x = state["x"]
    y = state["y"]
    yaw = state["yaw"]
    v = state["v"]
    s = state["s"]
    ey = state["ey"]
    soc = state["soc"]

    throttle = np.clip(control["throttle"], 0.0, cfg["max_throttle"])
    steer = np.clip(control["steer"], 
                    -np.deg2rad(cfg["max_steer_deg"]), 
                    np.deg2rad(cfg["max_steer_deg"]))

    # ----------------------------
    # 2. Track reference geometry
    # ----------------------------
    x_ref, y_ref, psi_ref, curvature_ref, slope_ref = get_track_props(track, s)

    # heading difference between car and track
    yaw_error = yaw - psi_ref

    # ----------------------------
    # 3. Motor model & forces
    # ----------------------------
    # wheel angular speed
    wheel_omega = v / cfg["wheel_radius_m"]      

    # motor rpm
    motor_rpm = wheel_omega * cfg["gear_ratio"] * 9.549296585  # 60/(2π)

    # simple torque curve (placeholder)
    if motor_rpm < cfg["max_motor_rpm"]:
        torque = cfg["peak_torque_Nm"] * throttle * (1 - motor_rpm / cfg["max_motor_rpm"])
    else:
        torque = 0.0

    # drivetrain → force at wheels
    F_drive = (torque * cfg["gear_ratio"] * cfg["drivetrain_eff"]) / cfg["wheel_radius_m"]

    # aero drag
    F_drag = 0.5 * cfg["air_density"] * cfg["Cd"] * cfg["A"] * v**2

    # rolling resistance
    F_roll = cfg["mass_kg"] * 9.81 * cfg["Crr"]

    # slope resistance (gravity component)
    # slope_ref is Δz/Δs → approx = sin(grade)
    F_slope = cfg["mass_kg"] * 9.81 * slope_ref

    # total longitudinal acceleration
    F_total = F_drive - F_drag - F_roll - F_slope
    a = F_total / cfg["mass_kg"]

    # ----------------------------
    # 4. Update motion (bicycle model)
    # ----------------------------
    L = cfg["wheelbase_m"]

    # yaw rate
    yaw_dot = v / L * np.tan(steer)

    # update velocity
    v_next = max(0.0, v + a * dt)

    # update yaw
    yaw_dot = v / cfg["wheelbase_m"] * np.tan(steer)
    yaw_next = yaw + yaw_dot * dt


    # update global position
    x_next = x + v_next * np.cos(yaw_next) * dt
    y_next = y + v_next * np.sin(yaw_next) * dt

    # ----------------------------
    # 5. Update track-relative variables
    # ----------------------------

    # advance along track
    ds = v_next * np.cos(yaw_error) * dt
    s_next = s + ds

    # update lateral error
    dey = v_next * np.sin(yaw_error) * dt
    ey_next = ey + dey

    # ----------------------------
    # 6. Energy model & SOC update
    # ----------------------------
    # mechanical power at wheel
    P_mech = F_drive * v_next    

    # electrical power (simple model)
    if cfg["drivetrain_eff"] > 0:
        P_elec = P_mech / cfg["drivetrain_eff"]
    else:
        P_elec = 0

    # Wh consumed during dt
    energy_used_Wh = (P_elec * dt) / 3600.0

    soc_next = soc - (energy_used_Wh / cfg["battery_Wh"])
    soc_next = max(0.0, soc_next)
    energy_used_J = P_elec * dt

    # ----------------------------
    # 7. Return next state
    # ----------------------------
    return {
        "x": x_next,
        "y": y_next,
        "yaw": yaw_next,
        "v": v_next,
        "s": s_next,
        "ey": ey_next,
        "soc": soc_next,
        "energy_J": energy_used_J,   # NEW
        "slope": slope_ref,          # NEW
    }


def run_simulation(track, cfg,
                   throttle_command=0.4,
                   k_steer=2.0,
                   dt=0.05):
    """
    Runs a full 4-lap simulation using the 2D bicycle model.
    Returns the full log and final state.
    """

    # ------------------------------------
    # INITIAL STATE
    # ------------------------------------
    state = {
        "x": track.x_m[0],
        "y": track.y_m[0],
        "yaw": track.heading_rad[0],
        "v": 0.0,
        "s": 0.0,
        "ey": 0.0,
        "soc": 1.0,
    }

    # total distance for 4 laps
    target_distance = 4 * track.length_m

    # max time = 30 minutes (SEM rules)
    max_time = 1800.0

    # ------------------------------------
    # LOGGING
    # ------------------------------------
    log = {
        "time": [],
        "x": [],
        "y": [],
        "yaw": [],
        "v": [],
        "s": [],
        "ey": [],
        "soc": [],
        "throttle": [],
        "steer": [],
        "slope": [],
        "energy_J": [],        # per-step energy
        "energy_J_total": [],  # NEW: cumulative energy
    }

    cumulative_energy_J = 0.0   # NEW

    # ------------------------------------
    # SIMULATION LOOP
    # ------------------------------------
    t = 0.0
    while t < max_time and state["s"] < target_distance and state["soc"] > 0:

        # track reference based on current position
        _, _, psi_ref, _, slope_ref = get_track_props(track, state["s"])

        # use MPC (now passing t_now)
        control = mpc_solve_step(state, track, cfg, t_now=t, dt=dt, H=20, K=1000)

        # propagate dynamics
        state = simulate_vehicle_step(state, control, track, cfg, dt)

        # update cumulative energy
        cumulative_energy_J += state["energy_J"]

        # ------------------------------------
        # LOG DATA
        # ------------------------------------
        log["time"].append(t)
        log["x"].append(state["x"])
        log["y"].append(state["y"])
        log["yaw"].append(state["yaw"])
        log["v"].append(state["v"])
        log["s"].append(state["s"])
        log["ey"].append(state["ey"])
        log["soc"].append(state["soc"])
        log["throttle"].append(control["throttle"])
        log["steer"].append(control["steer"])
        log["slope"].append(state["slope"])
        log["energy_J"].append(state["energy_J"])
        log["energy_J_total"].append(cumulative_energy_J)   # NEW

        # increment time
        t += dt

    # ------------------------------------
    # SUMMARY OUTPUT
    # ------------------------------------
    print("=== Simulation Complete ===")
    print(f"Time elapsed: {t:.1f} sec")
    print(f"Distance traveled: {state['s']:.1f} m")
    print(f"Laps completed: {state['s']/track.length_m:.2f}")
    print(f"SOC remaining: {state['soc']*100:.2f}%")
    print(f"Total energy used: {cumulative_energy_J:.1f} J")  # optional

    return log, state


# ==========================================
# ======= Random Shooting MPC ==============
# ==========================================

def mpc_cost(traj, control_seq, track, cfg, t_now, dt):
    J = 0.0

    # speed bounds (convert km/h to m/s)
    v_soft_min = speed['v_soft_min']     # 30 km/h
    v_soft_max = speed['v_soft_max']    # 40 km/h

    v_hard_min = speed['v_hard_min']     # 20 km/h
    v_hard_max = speed['v_hard_max']    # 50 km/h

    BIG_PENALTY = 1e6
    MEDIUM_PENALTY = 2e4

    v_values = []

    for k in range(len(traj)):
        st = traj[k]
        u = control_seq[k]

        # ===========================
        # SPEED COST
        # ===========================
        v = max(0.0, st["v"])
        v_values.append(v)

        # 1. Hard speed bounds (super strict)
        if v < v_hard_min or v > v_hard_max:
            J += BIG_PENALTY

        # 2. Soft speed bounds (inside 30–40 km/h)
        elif v < v_soft_min:
            # slower than 30 → penalty gets worse as you drop
            J += MEDIUM_PENALTY * (v_soft_min - v)**2

        elif v > v_soft_max:
            # faster than 40 → penalty gets worse as you exceed
            J += MEDIUM_PENALTY * (v - v_soft_max)**2

        # 3. Incentive for accelerating early
        if st["s"] < 10.0:  
            # reward higher speeds in first 10 meters
            J -= 200.0 * v

        # ===========================
        # TRACK HEADING ERROR
        # ===========================
        psi_ref = np.interp(st["s"] % track.length_m, track.s_m, track.heading_rad)
        psi_err = st["yaw"] - psi_ref
        J += 6.0 * psi_err**2

        # ===========================
        # LATERAL ERROR
        # ===========================
        J += 12.0 * st["ey"]**2

        # ===========================
        # ENERGY SURROGATE
        # ===========================
        # encourage slower speeds inside band (efficiency)
        J += 0.2 * (v**2)

        # ===========================
        # CONTROL SMOOTHNESS
        # ===========================
        if k > 0:
            du_t = control_seq[k][0] - control_seq[k-1][0]
            du_s = control_seq[k][1] - control_seq[k-1][1]
            J += 2.0 * (du_t**2 + du_s**2)

        # ==========================================================
        # PROGRESS PENALTY — must reach certain distance by time
        # ==========================================================
        L = track.length_m
        total_distance = 4 * L
        total_time = 1800.0  # 30 min

        v_avg_required = total_distance / total_time  # m/s

        t_k = t_now + k * dt  # <-- FIXED HERE

        s_target = v_avg_required * t_k

        margin = 5.0  # meters allowed behind

        if st["s"] < s_target - margin:
            J += 3e5 * (s_target - st["s"])**2

        # ===========================
        # WRONG-WAY PENALTY
        # ===========================
        # car heading should align with track heading
        yaw_err = psi_err  # already computed
        # if cos(yaw_err) < 0 → pointing backwards
        if np.cos(yaw_err) < 0:
            J += BIG_PENALTY * 50   # extremely heavy penalty

        # car progress should always be forward
        ds_pred = st["v"] * np.cos(yaw_err) * dt
        if ds_pred < -0.1:  # going backwards > 0.1 m
            J += BIG_PENALTY * 50




    # ===================================================================
    # GLOBAL PREDICTION PENALTY (mean speed must stay above threshold)
    # ===================================================================
    avg_v = np.mean(v_values)

    if avg_v < v_soft_min * 0.9:   # below ~27 km/h average
        J += 5e5

    return J


def mpc_solve_step(state, track, cfg, t_now, dt=0.1, H=20, K=1000):
    best_cost = float("inf")
    best_action = (0.0, 0.0)

    throttle_max = cfg["max_throttle"]
    steer_max = np.deg2rad(cfg["max_steer_deg"])

    for _ in range(K):

        # sample a random control sequence
        control_seq = []
        for t in range(H):
            thr = np.random.uniform(0, throttle_max)
            ste = np.random.uniform(-steer_max, steer_max)
            control_seq.append((thr, ste))

        # simulate forward this sequence
        st = state.copy()
        traj = []
        for (thr, ste) in control_seq:
            control = {"throttle": thr, "steer": ste}
            st = simulate_vehicle_step(st, control, track, cfg, dt)
            traj.append(st)

        # evaluate cost
        J = mpc_cost(traj, control_seq, track, cfg, t_now, dt=dt)

        # update best
        if J < best_cost:
            best_cost = J
            best_action = control_seq[0]

    return {"throttle": best_action[0], "steer": best_action[1]}


def run_mpc_simulation(track, cfg, dt=0.1, debug_step=60):
    state = {
        "x": track.x_m[0],
        "y": track.y_m[0],
        "yaw": track.heading_rad[0],
        "v": 0.0,
        "s": 0.0,
        "ey": 0.0,
        "soc": 1.0,
        "slope": 0.0,
        "energy_J": 0.0,   # NEW: avoid KeyError in early prints
    }

    total_distance = 4 * track.length_m
    t = 0.0
    max_time = 1800.0  # 30 minutes safety timeout

    log = {
        "time": [], "x": [], "y": [], "v": [], "s": [], "yaw": [],
        "ey": [], "soc": [], "throttle": [], "steer": [],
        "slope": [], "energy_J": [], "energy_J_total": []  # NEW key
    }

    cumulative_energy_J = 0.0   # running total J

    while state["s"] < total_distance:

        # ⚠️ safety warnings (NOT stopping conditions)
        if t > max_time:
            print(f"⚠️ WARNING t={t:.1f}s: exceeded 30 min, forcing limp mode.")
        if state["soc"] <= 0:
            print(f"⚠️ WARNING t={t:.1f}s: SOC depleted, using limp throttle.")
            limp_throttle = 0.05
            control = {"throttle": limp_throttle, "steer": 0.0}
        else:
            # MPC chooses the next control
            control = mpc_solve_step(state, track, cfg, t_now=t, dt=dt, H=20, K=1000)

        # 1) simulate forward one step
        state = simulate_vehicle_step(state, control, track, cfg, dt)

        # 2) update cumulative energy
        cumulative_energy_J += state["energy_J"]

        # 3) log everything
        log["time"].append(t)
        log["x"].append(state["x"])
        log["y"].append(state["y"])
        log["v"].append(state["v"])
        log["s"].append(state["s"])
        log["yaw"].append(state["yaw"])
        log["ey"].append(state["ey"])
        log["soc"].append(state["soc"])
        log["throttle"].append(control["throttle"])
        log["steer"].append(control["steer"])
        log["slope"].append(state["slope"])
        log["energy_J"].append(state["energy_J"])
        log["energy_J_total"].append(cumulative_energy_J)

        # 4) debug every debug_step (AFTER update so energy/slope are valid)
        if int(t/dt) % debug_step == 0:
            print(
                f"[t={t:.1f}s] "
                f"v={state['v']:.2f}m/s, "
                f"slope={state['slope']:.3f}, "
                f"E_step={state['energy_J']:.1f}J, "
                f"E_tot={cumulative_energy_J:.1f}J, "
                f"s={state['s']:.1f}, ey={state['ey']:.2f}, soc={state['soc']:.3f}"
            )

        t += dt

    print("\nMPC Simulation finished:")
    print(f"  Time:              {t:.1f} sec (may exceed 30 min)")
    print(f"  SOC remaining:     {state['soc']*100:.1f}%")
    print(f"  Laps completed:    {state['s']/track.length_m:.2f}")
    print(f"  Total energy used: {cumulative_energy_J:.1f} J")

    return log, state

