import pandas as pd

# Load the CSV file
file_path = 'path_to_your_file.csv'  # Update this with the actual path
data = pd.read_csv(file_path, delimiter='\t')

# Extract rows for frameID 274 and frameID 273
frame_274 = data[data['Frame ID'] == 274]
frame_273 = data[data['Frame ID'] == 273]

# Extract and parse QR 2D corner points
corner_points_274 = eval(frame_274['QR 2D (Corner Points)'].values[0])
corner_points_273 = eval(frame_273['QR 2D (Corner Points)'].values[0])

# Compute positional differences (assuming the first corner point for simplicity)
delta_x = corner_points_273[0][0] - corner_points_274[0][0]
delta_y = corner_points_273[0][1] - corner_points_274[0][1]

# Extract and compute angle differences
yaw_274 = frame_274['Yaw (degrees)'].values[0]
yaw_273 = frame_273['Yaw (degrees)'].values[0]
pitch_274 = frame_274['Pitch (degrees)'].values[0]
pitch_273 = frame_273['Pitch (degrees)'].values[0]
roll_274 = frame_274['Roll (degrees)'].values[0]
roll_273 = frame_273['Roll (degrees)'].values[0]

delta_yaw = yaw_273 - yaw_274
delta_pitch = pitch_273 - pitch_274
delta_roll = roll_273 - roll_274

# Generate movement commands
commands = []

if delta_x > 0:
    commands.append('right')
elif delta_x < 0:
    commands.append('left')

if delta_y > 0:
    commands.append('up')
elif delta_y < 0:
    commands.append('down')

# Consider forward and backward based on distance or additional parameters if available
distance_274 = frame_274['Distance'].values[0]
distance_273 = frame_273['Distance'].values[0]

if distance_273 > distance_274:
    commands.append('forward')
elif distance_273 < distance_274:
    commands.append('backward')

# Consider camera angle adjustments
if delta_yaw > 0:
    commands.append('turn-right')
elif delta_yaw < 0:
    commands.append('turn-left')

if delta_pitch != 0:
    commands.append(f'adjust-pitch {delta_pitch}')

if delta_roll != 0:
    commands.append(f'adjust-roll {delta_roll}')

print("Commands to move from frame 274 to 273:", commands)
