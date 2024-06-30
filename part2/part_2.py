import pandas as pd

def safe_eval(expr):
    try:
        return eval(expr)
    except:
        return None


def extract_commands(data, frame2, frame1):

    # Extract rows for frameID 274 and frameID 273
    try:
        frame_2 = data[data['Frame ID'] == frame2]
        frame_1 = data[data['Frame ID'] == frame1]

        # Extract and parse QR 2D corner points
        corner_points_2 = safe_eval(frame_2['QR 2D (Corner Points)'].values[0])
        corner_points_1 = safe_eval(frame_1['QR 2D (Corner Points)'].values[0])

        if not corner_points_2 or not corner_points_1:
            raise ValueError("Invalid corner points data")

        # Compute positional differences (assuming the first corner point for simplicity)
        delta_x = corner_points_1[0][0] - corner_points_2[0][0]
        delta_y = corner_points_1[0][1] - corner_points_2[0][1]

        # Extract and compute angle differences
        yaw_2 = frame_2['Yaw (degrees)'].values[0]
        yaw_1 = frame_1['Yaw (degrees)'].values[0]
        pitch_2 = frame_2['Pitch (degrees)'].values[0]
        pitch_1 = frame_1['Pitch (degrees)'].values[0]
        roll_2 = frame_2['Roll (degrees)'].values[0]
        roll_1 = frame_1['Roll (degrees)'].values[0]

        delta_yaw = yaw_1 - yaw_2
        delta_pitch = pitch_1 - pitch_2
        delta_roll = roll_1 - roll_2

        # Generate movement commands with magnitude
        commands = []

        if delta_x > 0:
            commands.append(f'left { abs(delta_x)}')
        elif delta_x < 0:
            commands.append(f'right { abs(delta_x) }')

        if delta_y > 0:
            commands.append(f'up { abs(delta_y) }')
        elif delta_y < 0:
            commands.append(f'down {abs(delta_y)}')

        # Consider forward and backward based on distance or additional parameters if available
        distance_2 = frame_2['Distance'].values[0]
        distance_1 = frame_1['Distance'].values[0]
        delta_distance = distance_1 - distance_2

        if delta_distance > 0:
            commands.append(f'backward {abs(delta_distance)}')
        elif delta_distance < 0:
            commands.append(f'forward { abs(delta_distance)}')

        # Consider camera angle adjustments
        if delta_yaw > 0: 
            commands.append(f'turn-left {abs(delta_yaw)}')
        elif delta_yaw < 0:
            commands.append(f'turn-right {abs(delta_yaw)}')

        # if delta_pitch != 0:
        #     commands.append(f'adjust-pitch {delta_pitch}')

        # if delta_roll != 0:
        #     commands.append(f'adjust-roll {delta_roll}')

        return commands

    except KeyError as e:
        print(f"Column not found: {e}")
    except ValueError as e:
        print(f"Data error: {e}")



if __name__ == "__main__":

    # Load the CSV file
    file_path = 'outputs/challengeB_data.csv'  # Update this with the actual path
    data = pd.read_csv(file_path, sep=None, engine='python')

    # frame_1 = 274
    # frame_2 = 287
    # commands = extract_commands(data, frame_1, frame_2)
    # print(f"{frame_2} -> {frame_1}:", commands)

    # frame_1 = 273
    # frame_2 = 274
    # commands = extract_commands(data, frame_1, frame_2)
    # print(f"{frame_2} -> {frame_1}:", commands)


    for i in range(399, 386, -1):
        frame_1 = i - 1
        frame_2 = i

        commands = extract_commands(data, frame_2, frame_1)
        print(f"{frame_2} -> {frame_1}:", commands)


