car_config = {
    # Geometry
    "mass_kg": 90.0,
    "wheel_radius_m": 0.1778,
    "gear_ratio": 5.0,
    "wheelbase_m": 1.2,       # you can adjust later
    "max_steer_deg": 25.0,

    # Motor / drivetrain
    "max_motor_rpm": 2500.0,
    "peak_torque_Nm": 10.0,   # placeholder
    "drivetrain_eff": 0.85,
    "max_throttle": 0.8,

    # Aero / rolling
    "Cd": 0.2,
    "A": 0.3,
    "Crr": 0.003,
    "air_density": 1.225,

    # Battery
    "battery_Wh": 960.0,
}

# full 2D bicycle model state
state = {
    "x": 0.0,          # world x [m]
    "y": 0.0,          # world y [m]
    "yaw": 0.0,        # heading [rad]
    "v": 0.0,          # speed [m/s]
    "s": 0.0,          # distance along track [m]
    "ey": 0.0,         # lateral deviation from track centerline [m]
    "soc": 1.0,        # battery state-of-charge (0–1)
}

control = {
    "throttle": 0.0,   # 0 to max_throttle
    "steer": 0.0       # steering command [rad]
}

speed = {
    # speed bounds (convert km/h to m/s)
    'v_soft_min': 8.3,     # 30 km/h
    'v_soft_max': 11.1,    # 40 km/h

    'v_hard_min': 5.6,     # 20 km/h
    'v_hard_max': 13.9,    # 50 km/h
}