# -*- coding: utf-8 -*-
"""
MPC for four laps on Lusail with hard speed bounds 30–60 km/h
- Minimizes energy-per-meter with a schedule-aware progress reward to finish within 30 minutes
- Uses throttle cap 0.8 (as per your hardware)
- Falls back to a simple controller if IPOPT fails transiently
Requirements: casadi, numpy, scipy
"""

import json
import math
import numpy as np
import casadi as ca
from scipy.interpolate import splprep, splev
import pathlib
import csv

# ===== USER / VEHICLE PARAMETERS =====
GEOJSON_PATH = "qa-2004.geojson"

# Mass: driver 55 kg + car 35 kg
m = 55.0 + 35.0
g = 9.81

# Wheel radius (14 in RADIUS assumed; if 14-in DIAMETER, set r_w=0.1778)
r_w = 0.3556
Lb  = 1.6            # wheelbase (m)  TODO: update when measured
CdA = 0.30           # m^2           TODO: refine with coastdown
Crr = 0.004          # -             TODO: refine with coastdown
rho = 1.18           # kg/m^3 Doha sea-level warm
eta_drive = 0.90     # drivetrain efficiency
gear_ratio = 5.0     # motor rpm = gear_ratio * wheel rpm

# Simple DC-like motor map placeholders
omega_no_load = 2500.0 * 2*math.pi/60.0   # rad/s (effective)
T_stall = 8.0                              # N·m @ motor shaft (placeholder)

# Speed limits (HARD bounds)
v_min_kmh = 30.0
v_max_kmh = 60.0
v_min = v_min_kmh / 3.6
v_max = v_max_kmh / 3.6

# Throttle limits (hardware cap is 80%)
u_th_min, u_th_max = 0.0, 0.8

# Lap/time goals
TARGET_LAPS = 4
TIME_LIMIT_SEC = 30 * 60              # 30 minutes
ENERGY_BUDGET_J = 960.0 * 3600.0      # 960 Wh -> Joules (optional, not enforced here)

# ===== TRACK utilities =====
def lonlat_to_local_xy(lon, lat, lon0=None, lat0=None):
    R = 6371000.0
    if lon0 is None: lon0 = lon[0]
    if lat0 is None: lat0 = lat[0]
    lon = np.asarray(lon); lat = np.asarray(lat)
    x = np.deg2rad(lon - lon0) * R * math.cos(math.radians(lat0))
    y = np.deg2rad(lat - lat0) * R
    return x, y, lon0, lat0

def arclength(x, y):
    dx = np.diff(x); dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    return np.concatenate(([0.0], np.cumsum(ds)))

def fit_centerline_spline(x, y, smooth=0.5):
    tck, _ = splprep([x, y], s=smooth, k=3)
    return tck

def eval_centerline(tck, s_query, s_total):
    u = (s_query % s_total) / s_total
    xy = splev(u, tck)
    dxy = splev(u, tck, der=1)
    ddxy = splev(u, tck, der=2)
    x, y = xy[0], xy[1]
    dx, dy = dxy[0], dxy[1]
    ddx, ddy = ddxy[0], ddxy[1]
    psi_c = np.arctan2(dy, dx)
    denom = (dx*dx + dy*dy)**1.5 + 1e-9
    kappa = (dx*ddy - dy*ddx) / denom
    return x, y, psi_c, kappa

# ===== Load track =====
print(f"Loading {GEOJSON_PATH}...")
if not pathlib.Path(GEOJSON_PATH).exists():
    print(f"ERROR: {GEOJSON_PATH} not found!")
    print("Download (example source): https://raw.githubusercontent.com/bacinger/f1-circuits/master/circuits/qa-2004.geojson")
    raise SystemExit(1)

with open(GEOJSON_PATH, "r") as f:
    gj = json.load(f)

coords = gj["features"][0]["geometry"]["coordinates"]
lon = [c[0] for c in coords]
lat = [c[1] for c in coords]
x_map, y_map, _, _ = lonlat_to_local_xy(lon, lat)
s_arr = arclength(x_map, y_map)
s_total = float(s_arr[-1])

print(f"Track length: {s_total:.1f} m")
print(f"Speed bounds: {v_min_kmh:.1f}–{v_max_kmh:.1f} km/h ({v_min:.2f}–{v_max:.2f} m/s)")
print(f"Goal: {TARGET_LAPS} laps within {TIME_LIMIT_SEC/60:.1f} minutes\n")

# Spline + curvature
M = 1600
s_grid = np.linspace(0, s_total, M)
tck = fit_centerline_spline(x_map, y_map, smooth=0.5)
xg, yg, psig, kappag = eval_centerline(tck, s_grid, s_total)
kappa_interp_fn = ca.interpolant('kappa', 'linear', [s_grid], kappag)

# ===== Vehicle models =====
def motor_torque_from_throttle_cas(omega_m, u_th):
    # Linear torque-speed; clip at zero
    T = T_stall * (1.0 - omega_m / (omega_no_load - 1e-6))
    return ca.fmax(0.0, T) * u_th

def power_and_accel_cas(v, u_th, omega_w):
    omega_m = omega_w * gear_ratio
    T_m = motor_torque_from_throttle_cas(omega_m, u_th)
    P_mech = T_m * omega_m
    # Wheel force via reduction
    F_trac = (T_m * gear_ratio * eta_drive) / (r_w + 1e-9)
    F_aero = 0.5 * rho * CdA * v * v
    F_roll = Crr * m * g
    a = (F_trac - F_aero - F_roll) / m
    P_elec = P_mech / eta_drive
    return P_elec, a

def motor_torque_from_throttle_num(omega_m, u_th):
    T = T_stall * (1.0 - omega_m / (omega_no_load - 1e-6))
    return max(0.0, T) * u_th

def power_and_accel_num(v, u_th, omega_w):
    omega_m = omega_w * gear_ratio
    T_m = motor_torque_from_throttle_num(omega_m, u_th)
    P_mech = T_m * omega_m
    F_trac = (T_m * gear_ratio * eta_drive) / (r_w + 1e-9)
    F_aero = 0.5 * rho * CdA * v * v
    F_roll = Crr * m * g
    a = (F_trac - F_aero - F_roll) / m
    P_elec = P_mech / eta_drive
    return float(P_elec), float(a)

# ===== MPC setup =====
nx, nu = 5, 2                  # [s, e_y, e_psi, v, delta], [delta_dot, u_th]
N = 20                         # horizon steps
dt = 0.2                       # 4.0 s horizon

# Weights (tune as needed)
Q_e_y   = 1.0
Q_e_psi = 0.5
R_delta = 1e-4
R_u     = 1e-3
W_energy = 0.6                 # J/m weight
e_y_max = 4.0
delta_max = np.deg2rad(35.0)
delta_rate_max = np.deg2rad(120.0)

# Progress weighting (schedule-aware)
# This scales up automatically if you are "behind schedule".
BASE_PROGRESS_W = 0.02         # starting progress reward (N·s/m units)
eps_v = 0.3

opti = ca.Opti()
X = opti.variable(nx, N+1)
U = opti.variable(nu, N)

s0     = opti.parameter()
e_y0   = opti.parameter()
e_psi0 = opti.parameter()
v0     = opti.parameter()
delta0 = opti.parameter()

opti.subject_to(X[:,0] == ca.vertcat(s0, e_y0, e_psi0, v0, delta0))

# “behind schedule” scalar passed in from the supervisor each step
progress_W = opti.parameter()

J = 0
for k in range(N):
    xk = X[:,k]
    uk = U[:,k]
    s_k     = xk[0]; e_yk   = xk[1]; e_psik = xk[2]; v_k = xk[3]; delta_k = xk[4]
    delta_dot = uk[0]; u_th = uk[1]

    # track curvature
    kappa_k = kappa_interp_fn(ca.fmod(s_k, s_total))

    # curvilinear kinematics
    e_y_dot   = v_k * ca.sin(e_psik)
    e_psi_dot = (v_k / Lb) * ca.tan(delta_k) - v_k * kappa_k
    s_dot     = v_k * ca.cos(e_psik) / (1 - e_yk * kappa_k + 1e-9)

    omega_w = v_k / (r_w + 1e-9)
    P_elec_k, a_x = power_and_accel_cas(v_k, u_th, omega_w)

    # Euler integration
    s_next     = s_k     + dt * s_dot
    e_y_next   = e_yk    + dt * e_y_dot
    e_psi_next = e_psik  + dt * e_psi_dot
    v_next     = v_k     + dt * a_x
    delta_next = delta_k + dt * delta_dot

    xk1 = ca.vertcat(s_next, e_y_next, e_psi_next, v_next, delta_next)

    # Cost: energy per meter + tracking + smooth inputs - progress reward
    J += W_energy * (P_elec_k / (v_next + eps_v))
    J += Q_e_y * e_y_next**2 + Q_e_psi * e_psi_next**2
    J += R_delta * delta_dot**2 + R_u * u_th**2
    # progress encouragement (schedule-aware)
    J += -progress_W * (s_next - s_k)

    # Dynamics & constraints
    opti.subject_to(X[:,k+1] == xk1)
    # Inputs
    opti.subject_to(u_th_min <= U[1,k])
    opti.subject_to(U[1,k]    <= u_th_max)
    opti.subject_to(ca.fabs(U[0,k]) <= delta_rate_max)
    # States: hard speed box, lateral, steering amplitude
    opti.subject_to(v_min <= X[3,k+1])
    opti.subject_to(X[3,k+1] <= v_max)
    opti.subject_to(ca.fabs(X[1,k+1]) <= e_y_max)
    opti.subject_to(ca.fabs(X[4,k+1]) <= delta_max)

opti.minimize(J)

p_opts = {"expand": False}
s_opts = {"max_iter": 800, "print_level": 0, "tol": 1e-4, "acceptable_tol": 5e-4}
opti.solver("ipopt", p_opts, s_opts)

# ===== Feedforward helpers for warm start =====
def steady_delta_from_kappa_arr(kappa):
    return np.clip(np.arctan(Lb * kappa), -delta_max, delta_max)

def steady_throttle_for_speed(v):
    # Solve F_trac = F_aero + F_roll -> required motor torque -> throttle
    F_aero = 0.5 * rho * CdA * v*v
    F_roll = Crr * m * g
    F_req  = F_aero + F_roll
    Tm_req = F_req * r_w / (gear_ratio * eta_drive)
    omega_w = v / (r_w + 1e-9)
    omega_m = omega_w * gear_ratio
    T_avail = max(0.0, T_stall * (1.0 - omega_m/(omega_no_load - 1e-6)))
    if T_avail <= 1e-9:
        return u_th_min
    return float(np.clip(Tm_req / T_avail, u_th_min, u_th_max))

def build_initial_guesses(state, v_ref):
    s_i, e_y_i, e_psi_i, v_i, delta_i = state
    # Predict s along horizon at roughly v_ref
    s_pred = s_i + v_ref * dt * np.arange(N+1)
    kappa_pred = np.interp(s_pred % s_total, s_grid, kappag)
    delta_ff = steady_delta_from_kappa_arr(kappa_pred)
    u_ff = steady_throttle_for_speed(v_ref)

    # Seed U: steer toward delta_ff with rate limits, throttle at u_ff
    U_init = np.zeros((nu, N))
    X_init = np.zeros((nx, N+1))
    X_init[:,0] = np.array(state)
    delta_now = delta_i
    for k in range(N):
        delta_tgt = float(delta_ff[min(k, len(delta_ff)-1)])
        d_delta = np.clip((delta_tgt - delta_now)/dt, -delta_rate_max, +delta_rate_max)
        U_init[0,k] = d_delta
        U_init[1,k] = u_ff

        # quick numeric rollout (approx)
        s_k, e_yk, e_psik, v_k = X_init[0,k], X_init[1,k], X_init[2,k], X_init[3,k]
        kappa_k = np.interp((s_k % s_total), s_grid, kappag)
        e_y_dot   = v_k * np.sin(e_psik)
        e_psi_dot = (v_k / Lb) * np.tan(delta_now) - v_k * kappa_k
        s_dot     = v_k * np.cos(e_psik) / (1 - e_yk * kappa_k + 1e-9)

        F_aero = 0.5 * rho * CdA * v_k*v_k
        F_roll = Crr * m * g
        F_req  = F_aero + F_roll
        omega_w = v_k / (r_w + 1e-9)
        omega_m = omega_w * gear_ratio
        T_avail = max(0.0, T_stall * (1.0 - omega_m/(omega_no_load - 1e-6)))
        Tm = T_avail * U_init[1,k]
        F_trac = (Tm * gear_ratio * eta_drive) / (r_w + 1e-9)
        a_x = (F_trac - F_req) / m

        X_init[0,k+1] = s_k + dt * s_dot
        X_init[1,k+1] = e_yk + dt * e_y_dot
        X_init[2,k+1] = e_psik + dt * e_psi_dot
        X_init[3,k+1] = np.clip(v_k + dt * a_x, v_min, v_max)
        delta_now = np.clip(delta_now + dt * d_delta, -delta_max, +delta_max)
        X_init[4,k+1] = delta_now
    return X_init, U_init

def solve_one_step_mpc(state, prog_w):
    opti.set_value(s0, state[0])
    opti.set_value(e_y0, state[1])
    opti.set_value(e_psi0, state[2])
    opti.set_value(v0, state[3])
    opti.set_value(delta0, state[4])
    opti.set_value(progress_W, prog_w)
    try:
        sol = opti.solve()
        Xsol = sol.value(X)
        Usol = sol.value(U)
        # warm-start next with a shift
        opti.set_initial(X, np.hstack([Xsol[:,1:], Xsol[:,-1:]]))
        opti.set_initial(U, np.hstack([Usol[:,1:], Usol[:,-1:]]))
        return Xsol, Usol, True
    except Exception:
        return None, None, False

# ===== Closed-loop simulation (receding horizon) =====
print("Starting closed-loop MPC simulation...\n")

dt_sim = dt
time_sim = 0.0
energy_total = 0.0
s_traveled = 0.0
laps_completed = 0
fail_count = 0
max_solver_fail = 6

# Start around mid speed
v_init = (v_min + v_max) / 2.0
state = np.array([0.0, 0.0, 0.0, v_init, 0.0])  # [s, e_y, e_psi, v, delta]

# Initial guess
X0, U0 = build_initial_guesses(state, v_init)
opti.set_initial(X, X0)
opti.set_initial(U, U0)

# Logging for CSV
log_rows = []
distance_goal = TARGET_LAPS * s_total

def schedule_progress_weight(distance_done, time_used):
    """Adaptive progress weight: push harder if behind schedule, relax if ahead."""
    remaining_dist = max(0.0, distance_goal - distance_done)
    remaining_time = max(1.0, TIME_LIMIT_SEC - time_used)
    # Required average speed to finish on time:
    v_req = remaining_dist / remaining_time
    # Convert to a weight bump when v_req approaches v_max
    # Map v_req in [v_min, v_max] -> multiplier ~ [1, 5]
    ratio = np.clip((v_req - v_min) / max(1e-6, (v_max - v_min)), 0.0, 1.0)
    return float(BASE_PROGRESS_W * (1.0 + 4.0*ratio))

while (s_traveled < distance_goal) and (time_sim < TIME_LIMIT_SEC):
    # schedule-aware progress weight
    prog_w = schedule_progress_weight(s_traveled, time_sim)

    Xsol, Usol, ok = solve_one_step_mpc(state, prog_w)
    if ok:
        u_steer_rate = float(Usol[0,0])
        u_throttle   = float(Usol[1,0])
        fail_count = 0
    else:
        fail_count += 1
        # Fallback: PD-like steer to curvature, throttle to mid-speed
        s_k = state[0] % s_total
        idx = int((s_k / s_total) * (len(kappag)-1))
        kappa_k = kappag[idx]
        delta_des = np.clip(Lb * kappa_k, -delta_max, delta_max)
        u_steer_rate = float(np.clip(4.0*(delta_des - state[4]), -delta_rate_max, +delta_rate_max))
        v_target = (v_min + v_max)/2.0
        u_throttle = float(np.clip(0.5 + 0.1*(v_target - state[3]), u_th_min, u_th_max))

        if fail_count >= max_solver_fail:
            print("Too many solver failures — using fallback thereafter.")
            opti.set_initial(X, X0)
            opti.set_initial(U, U0)
            max_solver_fail = 1_000_000

    # Numeric integrate one step
    s_k, e_yk, e_psik, v_k, delta_k = state
    s_norm = s_k % s_total
    idx = int((s_norm / s_total) * (len(kappag)-1))
    idx = min(idx, len(kappag)-1)
    kappa_k = kappag[idx]

    e_y_dot   = v_k * math.sin(e_psik)
    e_psi_dot = (v_k / Lb) * math.tan(delta_k) - v_k * kappa_k
    s_dot     = v_k * math.cos(e_psik) / (1 - e_yk * kappa_k + 1e-9)

    omega_w = v_k / (r_w + 1e-9)
    P_elec_step, a_x = power_and_accel_num(v_k, u_throttle, omega_w)

    s_next     = s_k + dt_sim * s_dot
    e_y_next   = e_yk + dt_sim * e_y_dot
    e_psi_next = e_psik + dt_sim * e_psi_dot
    v_next     = np.clip(v_k + dt_sim * a_x, v_min, v_max)
    delta_next = np.clip(delta_k + dt_sim * u_steer_rate, -delta_max, +delta_max)

    # lap counting & stats
    if s_next >= (laps_completed+1) * s_total:
        laps_completed += 1
    step_dist = max(0.0, s_next - s_k)
    s_traveled += step_dist
    energy_total += max(0.0, P_elec_step) * dt_sim

    # Log
    log_rows.append([
        float(s_norm),
        float(np.rad2deg(delta_next)),
        float(v_next),
        float(u_throttle),
        float(P_elec_step),
        float(np.rad2deg(delta_k))
    ])

    state = np.array([
        s_next,                                  # keep growing; we mod only for lookups
        float(np.clip(e_y_next, -e_y_max, e_y_max)),
        float(e_psi_next),
        float(v_next),
        float(delta_next)
    ])

    time_sim += dt_sim

    if len(log_rows) % 25 == 0:
        epm = energy_total / max(s_traveled, 1e-6)
        print(f"t={time_sim:6.1f}s | laps={laps_completed} | s={s_traveled:7.1f}m | "
              f"v={state[3]*3.6:5.1f}km/h | E/m={epm:6.2f} J/m")

# ===== Results =====
out_csv = pathlib.Path("mpc_log_lusail.csv")
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["s_m_mod", "delta_deg_next", "v_mps_next", "u_throttle", "P_elec_W", "delta_deg_curr"])
    writer.writerows(log_rows)

print("\n" + "="*70)
print("SIMULATION COMPLETE")
print("="*70)
print(f"Elapsed time:      {time_sim:.1f} s ({time_sim/60:.2f} min)")
print(f"Distance traveled: {s_traveled:.1f} m  (goal {distance_goal:.1f} m)")
print(f"Laps completed:    {s_traveled/s_total:.2f}")
print(f"Total energy:      {energy_total:.1f} J")
print(f"Avg energy/m:      {energy_total/max(s_traveled,1e-6):.2f} J/m")
if time_sim > 0:
    print(f"Avg speed:         {s_traveled/time_sim*3.6:.1f} km/h")
if s_traveled >= distance_goal and time_sim <= TIME_LIMIT_SEC:
    print("✅ Finished 4 laps within 30 minutes.")
elif s_traveled >= distance_goal:
    print("⚠️ Finished 4 laps but exceeded 30 minutes. Increase progress weight or tune.")
else:
    print("❌ Did not finish 4 laps. Increase progress weight or revise limits.")
print("="*70)
