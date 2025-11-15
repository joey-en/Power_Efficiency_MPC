from src.load_track import *
from src.simulate_car import *
from config import car_config

track = load_track_from_csv("sem_apme_2025-track_coordinates.csv")

# log, final_state = run_simulation(track, car_config)

log, final_state = run_mpc_simulation(track, car_config, debug_step=100)
animate_car_with_telemetry(log, track, car_config)
plot_simulation_metrics(log)
