import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Track:
    s_m: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    altitude_m: np.ndarray
    heading_rad: np.ndarray
    grade: np.ndarray
    curvature: np.ndarray
    length_m: float


def load_track_from_csv(
    file_path="/sem_apme_2025-track_coordinates.csv"
) -> Track:

    df = pd.read_csv(file_path, sep="\t")

    lat = df["latitude"].to_numpy(float)
    lon = df["longitude"].to_numpy(float)
    alt = df["altitude (m)"].to_numpy(float)
    s_km = df["distance (km)"].to_numpy(float)

    # distance in meters
    s_m = s_km * 1000.0

    # convert lat/lon to local x,y in meters
    R = 6371000.0
    lat0 = np.deg2rad(lat[0])
    lon0 = np.deg2rad(lon[0])

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    dlat = lat_rad - lat0
    dlon = lon_rad - lon0

    x_m = R * np.cos(lat0) * dlon
    y_m = R * dlat

    # compute heading
    dx = np.diff(x_m)
    dy = np.diff(y_m)
    segment_heading = np.arctan2(dy, dx)  # N-1

    heading = np.zeros_like(lat)
    heading[0] = segment_heading[0]
    heading[-1] = segment_heading[-1]

    if len(segment_heading) > 1:
        heading[1:-1] = 0.5 * (segment_heading[:-1] + segment_heading[1:])
    else:
        heading[1:-1] = segment_heading[0]

    # slope / grade
    ds = np.diff(s_m)
    dalt = np.diff(alt)

    ds_safe = np.where(np.abs(ds) < 1e-6, 1e-6, ds)  # avoid division by zero
    grade_seg = dalt / ds_safe  # N-1

    grade = np.zeros_like(alt)
    grade[0] = grade_seg[0]
    grade[-1] = grade_seg[-1]
    if len(grade_seg) > 1:
        grade[1:-1] = 0.5 * (grade_seg[:-1] + grade_seg[1:])
    else:
        grade[1:-1] = grade_seg[0]

    # curvature (✔ fixed)
    dheading = np.diff(heading)   # N-1
    curvature_seg = dheading / ds_safe  # N-1

    curvature = np.zeros_like(heading)
    curvature[0] = curvature_seg[0]
    curvature[-1] = curvature_seg[-1]
    if len(curvature_seg) > 1:
        curvature[1:-1] = 0.5 * (curvature_seg[:-1] + curvature_seg[1:])
    else:
        curvature[1:-1] = curvature_seg[0]

    length_m = float(s_m[-1] - s_m[0])

    return Track(
        s_m=s_m,
        x_m=x_m,
        y_m=y_m,
        lat_deg=lat,
        lon_deg=lon,
        altitude_m=alt,
        heading_rad=heading,
        grade=grade,
        curvature=curvature,
        length_m=length_m,
    )

def get_track_props(track, s):
    """
    Returns interpolated track centerline properties at distance s:
    x_ref, y_ref, heading_ref, curvature, slope
    """
    # wrap around track (loop)
    s = s % track.length_m 

    x = np.interp(s, track.s_m, track.x_m)
    y = np.interp(s, track.s_m, track.y_m)
    heading = np.interp(s, track.s_m, track.heading_rad)
    curvature = np.interp(s, track.s_m, track.curvature)
    grade = np.interp(s, track.s_m, track.grade)

    return x, y, heading, curvature, grade

# =========================================================
# ================== Plotting Functions ===================
# =========================================================

from mpl_toolkits.mplot3d import Axes3D
def plot_track(track: Track, show_start=True):

    plt.figure(figsize=(6, 6))
    plt.plot(track.x_m, track.y_m, linewidth=1.5)

    if show_start:
        plt.scatter(track.x_m[0], track.y_m[0], color="red")
        plt.text(track.x_m[0], track.y_m[0], " START", fontsize=8)

    plt.gca().set_aspect("equal", "box")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Lusail Track Centerline")
    plt.grid(True)
    plt.show()

def plot_track_altitude(track: Track):
    """
    Plot altitude vs distance along track.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(track.s_m, track.altitude_m, linewidth=1.5)
    plt.xlabel("Distance along track [m]")
    plt.ylabel("Altitude [m]")
    plt.title("Altitude Profile of Track")
    plt.grid(True)
    plt.show()


def plot_track_3d(track: Track):
    """
    3D visualization of track centerline using (x, y, altitude).
    """
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot3D(track.x_m, track.y_m, track.altitude_m, linewidth=1.5)

    # Mark start
    ax.scatter(track.x_m[0], track.y_m[0], track.altitude_m[0], color="red", s=40)
    ax.text(track.x_m[0], track.y_m[0], track.altitude_m[0], " START")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("Altitude [m]")
    ax.set_title("3D Track Visualization")
    plt.tight_layout()
    plt.show()


def animate_car_with_telemetry(log, track, cfg):
    plt.figure(figsize=(7,7))

    # Preload arrays for speed, slope, etc.
    s_values = np.array(log["s"])
    v_values = np.array(log["v"])
    throttle_values = np.array(log.get("throttle", [0]*len(s_values)))
    steer_values = np.array(log.get("steer", [0]*len(s_values)))

    # compute slope/grade (%) along the entire log
    slope_values = np.interp(s_values % track.length_m, track.s_m, track.grade) * 100.0

    for i in range(0, len(log["x"]), 50):  # skip frames for speed

        plt.clf()

        # --- Track centerline ---
        plt.plot(track.x_m, track.y_m, 'k--', linewidth=1, label="Track")

        # --- Car trajectory up to current time ---
        plt.plot(log["x"][:i], log["y"][:i], 'b-', linewidth=1.2, label="Car path")

        # --- Current car position ---
        plt.scatter(log["x"][i], log["y"][i], color='red', s=40)

        # --- Telemetry text block ---
        speed_kmh = v_values[i] * 3.6
        throttle_pct = throttle_values[i] * 100
        steer_deg = np.rad2deg(steer_values[i])
        slope_pct = slope_values[i]

        telemetry = (
            f"Time: {log['time'][i]:.1f} s\n"
            f"Speed: {speed_kmh:.1f} km/h\n"
            f"Throttle: {throttle_pct:.1f}%\n"
            f"Steer: {steer_deg:.1f}°\n"
            f"Slope: {slope_pct:.2f}%"
        )

        plt.text(
            0.02, 0.98, telemetry,
            transform=plt.gca().transAxes,
            fontsize=10, family='monospace',
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

        # --- Plot formatting ---
        plt.title("Car Simulation with Telemetry")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.gca().set_aspect("equal", "box")
        plt.grid(True)

        plt.pause(0.001)

    plt.show()
    
def plot_simulation_metrics(log):
    time = np.array(log["time"])
    v = np.array(log["v"])
    throttle = np.array(log["throttle"])
    steer = np.array(log["steer"])
    slope = np.array(log["slope"])

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # ---- 1. Speed ----
    axs[0].plot(time, v, label="Speed (m/s)", color='b')
    axs[0].set_ylabel("Speed (m/s)")
    axs[0].grid(True)
    axs[0].set_title("Vehicle Speed, Throttle, Steering, and Track Slope")

    # ---- 2. Throttle ----
    axs[1].plot(time, throttle, label="Throttle", color='g')
    axs[1].set_ylabel("Throttle")
    axs[1].set_ylim([-0.05, 1.0])
    axs[1].grid(True)

    # ---- 3. Steering ----
    axs[2].plot(time, steer, label="Steer (rad)", color='orange')
    axs[2].set_ylabel("Steer (rad)")
    axs[2].grid(True)

    # ---- 4. Slope (secondary axis) ----
    ax4 = axs[3]
    ax4.plot(time, slope, label="Slope (grade)", color='purple')
    ax4.set_ylabel("Slope (m/m)")
    ax4.grid(True)

    # Secondary y-axis with slope in percent
    ax4b = ax4.twinx()
    ax4b.plot(time, slope * 100, color='red', alpha=0.4)
    ax4b.set_ylabel("Slope (%)")

    axs[3].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()
