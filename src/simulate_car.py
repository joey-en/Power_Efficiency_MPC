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
    }

    # ------------------------------------
    # SIMULATION LOOP
    # ------------------------------------
    t = 0.0
    while t < max_time and state["s"] < target_distance and state["soc"] > 0:

        # track reference based on current position
        _, _, psi_ref, _, slope_ref = get_track_props(track, state["s"])

        # simple steering controller: track-heading following
        control = mpc_solve_step(state, track, cfg, dt=dt, H=10, K=300)

        # steer_cmd = k_steer * (psi_ref - state["yaw"])
        # # control input
        # control = {
        #     "throttle": throttle_command,
        #     "steer": steer_cmd
        # }

        # propagate dynamics
        state = simulate_vehicle_step(state, control, track, cfg, dt)

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
        log["throttle"].append(throttle_command)
        log["slope"].append(slope_ref)
        log.setdefault("throttle", []).append(control["throttle"])
        log.setdefault("steer", []).append(control["steer"])


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

    return log, state

# ==========================================
# ======= Random Shooting MPC ==============
# ==========================================

def mpc_cost(traj, control_seq, track, cfg):
    J = 0.0

    # speed limits in m/s
    v_min = speed["v_min"]   
    v_max = speed["v_max"] 

    for k in range(len(traj)):
        st = traj[k]
        u = control_seq[k]

        psi_ref = np.interp(st["s"] % track.length_m, track.s_m, track.heading_rad)
        psi_err = st["yaw"] - psi_ref

        v = st["v"]

        # ---------------------------------------------------
        # 🔥 ENERGY SURROGATE
        P = v * v
        J_energy = 1.0 * P / max(0.3, v)

        # ---------------------------------------------------
        # 🔥 LATERAL ERROR
        J_lat = 10.0 * (st["ey"] ** 2)

        # ---------------------------------------------------
        # 🔥 HEADING ERROR
        J_head = 5.0 * (psi_err ** 2)

        # ---------------------------------------------------
        # 🔥 SPEED-WINDOW PENALTY
        if v < v_min:
            # too slow (BIG penalty)
            J_speed = 50.0 * (v_min - v) ** 2
        elif v > v_max:
            # too fast (even BIGGER penalty)
            J_speed = 80.0 * (v - v_max) ** 2
        else:
            # in the sweet spot
            J_speed = 0.0

        # ---------------------------------------------------
        # 🔥 CONTROL SMOOTHNESS
        if k > 0:
            du_t = control_seq[k][0] - control_seq[k-1][0]
            du_s = control_seq[k][1] - control_seq[k-1][1]
            J_smooth = 2.0 * (du_t**2 + du_s**2)
        else:
            J_smooth = 0.0

        # ---------------------------------------------------
        # TOTAL COST
        J += J_energy + J_lat + J_head + J_speed + J_smooth

    return J


def mpc_solve_step(state, track, cfg, dt=0.1, H=10, K=300):
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
        J = mpc_cost(traj, control_seq, track, cfg)

        # update best
        if J < best_cost:
            best_cost = J
            best_action = control_seq[0]

    return {"throttle": best_action[0], "steer": best_action[1]}

def run_mpc_simulation(track, cfg, dt=0.1, debug_step=20):
    state = {
        "x": track.x_m[0],
        "y": track.y_m[0],
        "yaw": track.heading_rad[0],
        "v": 0.0,
        "s": 0.0,
        "ey": 0.0,
        "soc": 1.0,
    }

    total_distance = 4 * track.length_m
    t = 0.0
    max_time = 1800.0  # 30 minutes safety timeout

    log = { "time": [], "x": [], "y": [], "v": [], "s": [], "yaw": [], "ey": [], "soc": [], "throttle": [], "steer": [] }

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
            control = mpc_solve_step(state, track, cfg, dt=dt, H=10, K=200)

        # debug every debug_step
        if int(t/dt) % debug_step == 0:
            print(f"[t={t:.1f}s] MPC → thr={control['throttle']:.3f}, "
                  f"steer={np.rad2deg(control['steer']):.2f}°, "
                  f"v={state['v']:.2f} m/s, ey={state['ey']:.2f}, "
                  f"s={state['s']:.1f}, soc={state['soc']:.3f}")

        # simulate forward
        state = simulate_vehicle_step(state, control, track, cfg, dt)

        # log
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

        t += dt

    print("\nMPC Simulation finished:")
    print(f"  Time:              {t:.1f} sec (may exceed 30 min)")
    print(f"  SOC remaining:     {state['soc']*100:.1f}%")
    print(f"  Laps completed:    {state['s']/track.length_m:.2f}")

    return log, state
