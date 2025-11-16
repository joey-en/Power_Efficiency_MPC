from src.load_track import *
from src.simulate_car import *
from config import car_config
from src.utils import *

track = load_track_from_csv("sem_apme_2025-track_coordinates.csv")

# log, final_state = run_simulation(track, car_config)

log, final_state = run_mpc_simulation(track, car_config, debug_step=100)
save_log_to_json(log, "./results_wrongway_penalty_2/log.json")
generate_pace_notes(log, track, "./results_wrongway_penalty_2/pace_notes.txt")
export_mpc_csv(log, track, "./results_wrongway_penalty_2/driving_instructions.csv")
animate_car_with_telemetry(log, track, car_config)