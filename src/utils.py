import numpy as np
import csv,json

def save_log_to_json(log, out_file="results/log.json"):
    with open(out_file, 'w') as f:
        # 3. Use json.dump() to write the data to the file
        # 'indent' argument makes the JSON output more readable
        json.dump(log, f, indent=4)
    print(f"Data successfully saved to {out_file}")

def generate_pace_notes(log, track, out_file="results/pace_notes.txt"):
    """
    Convert raw MPC logs into human-readable driving instructions.
    """
    s_arr = np.array(log["s"])
    v_arr = np.array(log["v"])
    throttle_arr = np.array(log["throttle"])
    steer_arr = np.array(log["steer"])

    # compute curvature & slope
    curvature = np.interp(s_arr % track.length_m, track.s_m, track.curvature)
    slope = np.interp(s_arr % track.length_m, track.s_m, track.grade)

    # Open file for writing
    with open(out_file, "w") as f:
        f.write("=== Driving Instructions (Pace Notes) ===\n")
        f.write("Generated from MPC Simulation\n\n")

        segment_size = 20  # meters per instruction
        max_s = s_arr[-1]

        for start_s in np.arange(0, max_s, segment_size):
            # pick nearest index
            idx = (np.abs(s_arr - start_s)).argmin()

            speed = v_arr[idx] * 3.6
            thr = throttle_arr[idx] * 100
            st_deg = np.rad2deg(steer_arr[idx])
            curv = curvature[idx]
            slp = slope[idx] * 100

            # classify corner severity
            if abs(curv) < 0.002:
                corner = "Straight"
            elif abs(curv) < 0.008:
                corner = "Mild turn"
            else:
                corner = "Sharp turn"

            # classify slope
            if abs(slp) < 0.2:
                slope_desc = "Flat"
            elif slp > 0:
                slope_desc = "Uphill"
            else:
                slope_desc = "Downhill"

            f.write(
                f"Distance {start_s:.0f}–{start_s+segment_size:.0f} m:\n"
                f"  • {corner}, {slope_desc}\n"
                f"  • Recommended speed: {speed:.1f} km/h\n"
                f"  • Throttle: {thr:.1f}%\n"
                f"  • Steer: {st_deg:.1f}°\n\n"
            )

    print(f"Pace notes saved to {out_file}")

def export_mpc_csv(log, track, out_file="results/driving_instructions.csv"):
    """
    Save detailed MPC output to CSV:
    distance, speed, throttle, steer, slope, curvature
    """
    s_arr = np.array(log["s"])
    v_arr = np.array(log["v"])
    throttle_arr = np.array(log["throttle"])
    steer_arr = np.array(log["steer"])

    # interpolate slope & curvature
    curvature = np.interp(s_arr % track.length_m, track.s_m, track.curvature)
    slope = np.interp(s_arr % track.length_m, track.s_m, track.grade)

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "distance_m",
            "speed_mps",
            "speed_kph",
            "throttle_pct",
            "steer_deg",
            "slope_pct",
            "curvature"
        ])

        for i in range(len(s_arr)):
            writer.writerow([
                s_arr[i],
                v_arr[i],
                v_arr[i] * 3.6,
                throttle_arr[i] * 100,
                np.rad2deg(steer_arr[i]),
                slope[i] * 100,
                curvature[i]
            ])

    print(f"CSV instructions saved to {out_file}")
