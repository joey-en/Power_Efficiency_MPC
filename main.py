from src.load_track import *

track = load_track_from_csv("sem_apme_2025-track_coordinates.csv")

plot_track(track)               # 2D x–y
plot_track_altitude(track)      # 2D altitude vs distance
plot_track_3d(track)            # 3D track with terrain
