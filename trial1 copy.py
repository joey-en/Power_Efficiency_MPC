import json
import math
import numpy as np
import casadi as ca
from scipy.interpolate import splprep, splev
import pathlib

# ===== PARAMETERS =====
GEOJSON_PATH = "qa-2004.geojson"

m = 90.0
g = 9.81
r_w = 0.3556
Lb = 1.6
CdA = 0.30
Crr = 0.004
rho = 1.18
eta_drive = 0.90
gear_ratio = 5.0

omega_no_load = 2500.0 * 2*math.pi/60.0
T_stall = 8.0

v_target = 12.0          # m/s (~43 km/h)
u_th_cruise = 0.4        # cruise throttle
TIME_LIMIT_SEC = 30 * 60
TARGET_LAP_TIME = TIME_LIMIT_SEC

# Limits and MPC settings
v_min = 30/3.6
v_max = 40/3.6
u_th_min, u_th_max = 0.0, 1.0
delta_max = np.deg2rad(35.0)
delta_rate_max = np.deg2rad(120.0)

N = 8                   # short MPC horizon
dt = 0.2                # s
Q_e_y = 1.0
Q_e_psi = 0.5
Q_v = 0.1
R_delta = 1e-4
R_u = 1e-3
W_energy = 0.4
e_y_max = 4.0

# ===== TRACK utilities =====
def lonlat_to_local_xy(lon, lat, lon0=None, lat0=None):
    R = 6371000.0
    if lon0 is None:
        lon0 = lon[0]
    if lat0 is None:
        lat0 = lat[0]
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    x = np.deg2rad(lon - lon0) * R * math.cos(math.radians(lat0))
    y = np.deg2rad(lat - lat0) * R
    return x, y, lon0, lat0

def arclength(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    return np.concatenate(([0.0], np.cumsum(ds)))

def fit_centerline_spline(x, y, smooth=1.0):
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

# Load track
print(f"Loading {GEOJSON_PATH}...")
if not pathlib.Path(GEOJSON_PATH).exists():
    print(f"ERROR: {GEOJSON_PATH} not found!")
    print("Download: https://raw.githubusercontent.com/bacinger/f1-circuits/master/circuits/qa-2004.geojson")
    exit(1)

with open(GEOJSON_PATH) as f:
    gj = json.load(f)

coords = gj["features"][0]["geometry"]["coordinates"]
lon = [c[0] for c in coords]
lat = [c[1] for c in coords]
x_map, y_map, _, _ = lonlat_to_local_xy(lon, lat)
s_arr = arclength(x_map, y_map)
s_total = float(s_arr[-1])

print(f"Track length: {s_total:.1f} m | Target lap time: {TARGET_LAP_TIME/60:.1f} min\n")

# Spline + curvature
M = 800
s_grid = np.linspace(0, s_total, M)
tck = fit_centerline_spline(x_map, y_map, smooth=0.5)
xg, yg, psig, kappag = eval_centerline(tck, s_grid, s_total)
kappa_interp_fn = ca.interpolant('kappa', 'linear', [s_grid], kappag)

# ===== VEHICLE models (casadi and numeric) =====
def motor_torque_from_throttle_cas(omega_m, u_th):
    omega_frac = ca.fmin(omega_m / (omega_no_load + 1e-9), 1.0)
    return T_stall * (1.0 - omega_frac) * u_th

def power_and_accel_cas(v, u_th, omega_w):
    omega_m = omega_w * gear_ratio
    T_m = motor_torque_from_throttle_cas(omega_m, u_th)
    P_mech = T_m * omega_m
    F_trac = (T_m * gear_ratio * eta_drive) / (r_w + 1e-9)
    F_aero = 0.5 * rho * CdA * v * v
    F_roll = Crr * m * g
    F_net = F_trac - F_aero - F_roll
    a = F_net / m
    P_elec = P_mech / eta_drive
    return P_elec, a

def motor_torque_from_throttle_num(omega_m, u_th):
    omega_frac = min(omega_m / (omega_no_load + 1e-9), 1.0)
    return T_stall * (1.0 - omega_frac) * u_th

def power_and_accel_num(v, u_th, omega_w):
    omega_m = omega_w * gear_ratio
    T_m = motor_torque_from_throttle_num(omega_m, u_th)
    P_mech = T_m * omega_m
    F_trac = (T_m * gear_ratio * eta_drive) / (r_w + 1e-9)
    F_aero = 0.5 * rho * CdA * v * v
    F_roll = Crr * m * g
    F_net = F_trac - F_aero - F_roll
    a = F_net / m
    P_elec = P_mech / eta_drive
    return float(P_elec), float(a)

# ===== CASADI MPC setup =====
nx = 5
nu = 2

opti = ca.Opti()
X = opti.variable(nx, N+1)   # states: s, e_y, e_psi, v, delta
U = opti.variable(nu, N)     # inputs: delta_dot, u_th

s0 = opti.parameter()
e_y0 = opti.parameter()
e_psi0 = opti.parameter()
v0 = opti.parameter()
delta0 = opti.parameter()

opti.subject_to(X[:, 0] == ca.vertcat(s0, e_y0, e_psi0, v0, delta0))

J = 0
eps = 0.3

for k in range(N):
    xk = X[:, k]
    uk = U[:, k]
    # dynamics (casadi)
    s_k = xk[0]; e_yk = xk[1]; e_psik = xk[2]; v_k = xk[3]; delta_k = xk[4]
    delta_dot = uk[0]; u_th = uk[1]

    kappa_k = kappa_interp_fn(ca.fmod(s_k, s_total))
    e_y_dot = v_k * ca.sin(e_psik)
    e_psi_dot = (v_k / Lb) * ca.tan(delta_k) - v_k * kappa_k
    s_dot = v_k * ca.cos(e_psik) / (1 - e_yk * kappa_k + 1e-9)
    omega_w = v_k / (r_w + 1e-9)
    P_elec_k, a_x = power_and_accel_cas(v_k, u_th, omega_w)

    s_next = s_k + dt * s_dot
    e_y_next = e_yk + dt * e_y_dot
    e_psi_next = e_psik + dt * e_psi_dot
    v_next = v_k + dt * a_x
    delta_next = delta_k + dt * delta_dot

    xk1 = ca.vertcat(s_next, e_y_next, e_psi_next, v_next, delta_next)

    # cost
    J += W_energy * (P_elec_k / (v_next + eps))
    J += Q_e_y * e_y_next**2 + Q_e_psi * e_psi_next**2 + Q_v * (v_next - v_target)**2
    J += R_delta * delta_dot**2 + R_u * u_th**2

    # constraints
    opti.subject_to(X[:, k+1] == xk1)
    opti.subject_to(U[1, k] >= u_th_min)
    opti.subject_to(U[1, k] <= u_th_max)
    opti.subject_to(ca.fabs(U[0, k]) <= delta_rate_max)
    opti.subject_to(X[3, k+1] >= v_min)
    opti.subject_to(X[3, k+1] <= v_max)
    opti.subject_to(ca.fabs(X[1, k+1]) <= e_y_max)
    opti.subject_to(ca.fabs(X[4, k+1]) <= delta_max)

opti.minimize(J)

p_opts = {"expand": False}
s_opts = {"max_iter": 300, "print_level": 0, "tol": 1e-4, "acceptable_tol": 1e-3}
opti.solver("ipopt", p_opts, s_opts)

# initial warm-start guesses
X_init_guess = np.tile(np.array([0.0, 0.0, 0.0, v_target, 0.0])[:, None], (1, N+1))
U_init_guess = np.tile(np.array([0.0, u_th_cruise])[:, None], (1, N))
opti.set_initial(X, X_init_guess)
opti.set_initial(U, U_init_guess)

def solve_one_step_mpc(state):
    opti.set_value(s0, state[0])
    opti.set_value(e_y0, state[1])
    opti.set_value(e_psi0, state[2])
    opti.set_value(v0, state[3])
    opti.set_value(delta0, state[4])
    try:
        sol = opti.solve()
        Xsol = sol.value(X)
        Usol = sol.value(U)
        # warm-start for next iteration
        opti.set_initial(X, np.hstack([Xsol[:,1:], Xsol[:,-1:]]))
        opti.set_initial(U, np.hstack([Usol[:,1:], Usol[:,-1:]]))
        return Xsol, Usol, True
    except Exception:
        return None, None, False

# ===== SIMULATION loop (receding horizon) =====
print("Starting closed-loop MPC simulation (with fallback)...\n")
dt_sim = dt
time_sim = 0.0
energy_total = 0.0
s_traveled = 0.0
step_count = 0
laps_completed = 0
max_solver_fail = 6
fail_count = 0

state = np.array([0.0, 0.0, 0.0, v_target, 0.0])  # s, e_y, e_psi, v, delta

while time_sim < TARGET_LAP_TIME:
    Xsol, Usol, ok = solve_one_step_mpc(state)
    if ok:
        u_steer_rate = float(Usol[0,0])
        u_throttle = float(Usol[1,0])
        fail_count = 0
    else:
        fail_count += 1
        # fallback: simple P controllers
        s_k = state[0] % s_total
        idx = int((s_k / s_total) * (len(kappag)-1))
        kappa_k = kappag[idx]
        # steering approx
        delta_des = np.clip(Lb * kappa_k, -delta_max, delta_max)
        u_steer_rate = np.clip(4.0*(delta_des - state[4]), -delta_rate_max, delta_rate_max)
        # throttle to track v_target
        u_throttle = float(np.clip(u_th_cruise + 0.1*(v_target - state[3]), u_th_min, u_th_max))

        if fail_count >= max_solver_fail:
            print("Too many consecutive solver failures — switching to open-loop fallback for remainder.")
            # continue in fallback-only mode until end of simulation
            # set opti initial guesses to fallback to reduce future failures
            opti.set_initial(X, X_init_guess)
            opti.set_initial(U, U_init_guess)
            # but do not attempt solve in future iterations; force ok=False branch only
            # implement by setting a flag:
            max_solver_fail = 1_000_000

    # numeric integration (forward Euler) using numeric model
    s_k, e_yk, e_psik, v_k, delta_k = state
    # curvature at current s
    s_norm = s_k % s_total
    idx = int((s_norm / s_total) * (len(kappag)-1))
    kappa_k = kappag[idx]

    e_y_dot = v_k * math.sin(e_psik)
    e_psi_dot = (v_k / Lb) * math.tan(delta_k) - v_k * kappa_k
    s_dot = v_k * math.cos(e_psik) / (1 - e_yk * kappa_k + 1e-9)

    omega_w = v_k / (r_w + 1e-9)
    P_elec_step, a_x = power_and_accel_num(v_k, u_throttle, omega_w)

    s_next = s_k + dt_sim * s_dot
    e_y_next = e_yk + dt_sim * e_y_dot
    e_psi_next = e_psik + dt_sim * e_psi_dot
    v_next = v_k + dt_sim * a_x
    delta_next = delta_k + dt_sim * u_steer_rate

    # clamp/wrap and stats
    if s_next < s_k - s_total/2:
        s_next += s_total
        laps_completed += 1
    s_traveled += max(0.0, (s_next - s_k))
    energy_total += max(0.0, P_elec_step) * dt_sim

    state = np.array([
        s_next % (2*s_total),
        float(np.clip(e_y_next, -e_y_max, e_y_max)),
        float(e_psi_next),
        float(np.clip(v_next, v_min, v_max)),
        float(np.clip(delta_next, -delta_max, delta_max))
    ])

    time_sim += dt_sim
    step_count += 1

    if step_count % 25 == 0:
        epm = energy_total / max(s_traveled, 1e-6)
        print(f"t={time_sim:6.1f}s | s={state[0]:7.1f}m | v={state[3]:5.2f}m/s | e_y={state[1]:5.2f}m | E/m={epm:6.2f}J/m")

# final report
print("\n" + "="*70)
print("SIMULATION COMPLETE")
print("="*70)
print(f"Sim time:        {time_sim:.1f} s ({time_sim/60:.2f} min)")
print(f"Distance traveled:{s_traveled:.1f} m")
print(f"Laps completed:  {laps_completed + s_traveled//s_total:.1f}")
print(f"Total energy:    {energy_total:.1f} J")
print(f"Avg energy/m:    {energy_total/max(s_traveled,1e-6):.2f} J/m")
if time_sim > 0:
    print(f"Avg speed:       {s_traveled/time_sim*3.6:.1f} km/h")
print("="*70)