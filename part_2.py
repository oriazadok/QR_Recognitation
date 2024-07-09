import sys
import numpy as np
import pandas as pd
import cv2
import time

# Function to compute the 3D pose (distance, yaw, pitch, roll) from rotation and translation vectors
def compute_3d_pose(rvec, tvec):
    distance = np.linalg.norm(tvec)  # Calculate the distance
    R, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to rotation matrix
    yaw = np.arctan2(R[1, 0], R[0, 0])  # Calculate yaw angle
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))  # Calculate pitch angle
    roll = np.arctan2(R[2, 1], R[2, 2])  # Calculate roll angle

    return distance, yaw, pitch, roll

def calculate_live_camera_location(frame, camera_matrix, dist_coeffs):
    """
    Detects the ArUco marker in the given frame and returns its translation and rotation vectors.
    
    Parameters:
    frame (numpy.ndarray): The image frame containing the ArUco marker.
    camera_matrix (numpy.ndarray): Intrinsic camera matrix.
    dist_coeffs (numpy.ndarray): Distortion coefficients.
    
    Returns:
    The function returns camera_location, x_cam, y_cam, and z_cam, which describe the camera location and the orientation vectors relative to the ArUco marker.
    """

    corners, ids = detect_QR(frame)
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], 0.05, camera_matrix, dist_coeffs)
        distance, yaw, pitch, roll = compute_3d_pose(rvecs[0], tvecs[0])  # changed
        yaw, pitch, roll = np.degrees([yaw, pitch, roll])  # added
        return distance, yaw, pitch, roll
    else:
        return None, None, None, None

def get_movement_commands_for_this_frame(frame, next_frame_data, camera_matrix, dist_coeffs):
    """
    Calculates the best single movement command needed to move from the position and orientation of the ArUco 
    marker in the first frame to the position and orientation of the marker in the second frame.
    
    Parameters:
    frame1 (numpy.ndarray): The first image frame containing an ArUco marker.
    frame2 (numpy.ndarray): The second image frame containing an ArUco marker.
    camera_matrix (numpy.ndarray): Camera matrix obtained from camera calibration.
    dist_coeffs (numpy.ndarray): Distortion coefficients obtained from camera calibration.
    
    Returns:
    str: The best movement command to move from the first marker to the second.
    """
    # Detect ArUco markers in both frames
    
    live_dist, live_yaw, live_pitch, live_roll = calculate_live_camera_location(frame, camera_matrix, dist_coeffs)
    log_dist, log_yaw, log_pitch, log_roll = next_frame_data["Distance"].iloc[0], next_frame_data["Yaw (degrees)"].iloc[0], next_frame_data["Pitch (degrees)"].iloc[0], next_frame_data["Roll (degrees)"].iloc[0]

    corners_data = next_frame_data["QR 2D (Corner Points)"].iloc[0]

    # Convert the string representation of lists into actual Python lists
    list_of_lists = eval(corners_data)

    # Convert the list of lists into a NumPy array
    np_array = np.array(list_of_lists)

    log_x0 = float(np_array[0][0])
    log_y0 = float(np_array[0][1])

    log_id = next_frame_data.index.get_level_values('QR ID')[0]
    corners_all, ids = detect_QR(frame)
    corners = corners_all[0]
    for i in range (len(ids) - 1):
        if ids[i][0] == log_id:
            corners = corners_all[i]

    live_x0 = float(corners[0][0][0])
    live_y0 = float(corners[0][0][1])


    maxDistance =0
    diffs = dict()
    diffs['distance'] = log_dist - live_dist
    diffs['yaw'] = log_yaw - live_yaw
    diffs["vertical"] = live_y0 - log_y0
    diffs["horizontal"] = live_x0 - log_x0

    # print("log: ", log_yaw)
    # print("live: ", live_yaw)
    # print(diffs['distance'])
 
    command = 'move:\n'

    if diffs['yaw'].item() > 5:
        if abs(diffs['yaw'])> maxDistance:
            maxDistance = abs(diffs['yaw'])*5
            command = f'yaw: right\n'
            return command
    if diffs['yaw'].item() < -5:
        if abs(diffs['yaw'])> maxDistance:
            maxDistance = abs(diffs['yaw'])*5
            command = f'yaw: left\n'
            return command

    # forward backward condition
    if diffs['distance'].item() > 0.08:
        if abs(diffs['distance'])> maxDistance:
            maxDistance = abs(diffs['distance'])*250
            command = 'backward\n'
    if diffs['distance'].item() < -0.08:
        if abs(diffs['distance'])> maxDistance:
            maxDistance = abs(diffs['distance'])*250
            command = 'forward\n'

    # up down condition
    if diffs["vertical"] > 10:
        
        if abs(diffs['vertical'])> maxDistance:
            maxDistance = abs(diffs['vertical'])*2
            command = f'move down\n'
    if diffs["vertical"] < -10:
        if abs(diffs['vertical'])> maxDistance:
            maxDistance = abs(diffs['vertical'])*2
            command = f'move up\n'

    # left right condition
    if diffs["horizontal"] > 20:
        if abs(diffs['horizontal'])> maxDistance:
            maxDistance = abs(diffs['horizontal'])
            command = f'move right\n'
    if diffs["horizontal"] < -20:
        if abs(diffs['horizontal'])> maxDistance:
            maxDistance = abs(diffs['horizontal'])
            command = f'move left\n'

  
    return command



# get all rows from the csv with the same frame ID
def get_rows_by_frame_id(df, frame_id):
    """
    Retrieves all rows from a CSV file based on the frame ID.
    
    Parameters:
    file_path (str): The path to the CSV file.
    frame_id (int): The frame ID to search for.
    
    Returns:
    pandas.DataFrame: The rows matching the frame ID, or an empty DataFrame if the frame ID is not found.
    """
    matching_rows = df[df.index.get_level_values('Frame ID') == frame_id]
    if not matching_rows.empty:
        return matching_rows  # Return all rows that match the frame ID
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no matches are found

def calculate_next_frame(online_frame,df:pd.DataFrame):
    online_qrs = detect_QR(online_frame)[1]
    online_qrs = [x[0] for x in online_qrs]
    # Filter DataFrame to include only rows with QR IDs in the tuple
    df_filtered = df[df.index.get_level_values('QR ID').isin(online_qrs)]
    if df_filtered.empty:
        return None,None
    max_frame_id = df_filtered.index.get_level_values('Frame ID').max()
    max_frame_id_row = df_filtered.loc[(df_filtered.index.get_level_values('Frame ID') == max_frame_id)]
    corresponding_qr_id = max_frame_id_row.index.get_level_values('QR ID')
    return max_frame_id, corresponding_qr_id
    
# Function to detect QR codes in a gtarget_idiven frame
def detect_QR(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        # Detect markers using ArUco dictionary
        corners, ids, _ = cv2.aruco.detectMarkers(gray_frame, cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100))

        return corners, ids
    except Exception as e:
        print(f"Error detecting markers: {e}")
        return None, None

# Function to initialize camera parameters based on the resolution and field of view
def initialize_camera_parameters(resolution=(1280, 720), field_of_view=82.6):

    # Compute the focal length using the resolution and field of view
    focal_length = (resolution[0] / 2) / np.tan(np.deg2rad(field_of_view / 2))

    # Define the camera matrix
    camera_matrix = np.array([[focal_length, 0, resolution[0] / 2],
                              [0, focal_length, resolution[1] / 2],
                              [0, 0, 1]])
    # Initialize distortion coefficients to zero
    distortion_coeffs = np.zeros((4, 1))

    return camera_matrix, distortion_coeffs

def controller(file_path):
    # TODO add camera parameters 
    camera_matrix, dist_coeffs = initialize_camera_parameters()

    # Load the CSV file
    df = pd.read_csv(file_path, sep=None, engine='python')
    df.set_index(['Frame ID', 'QR ID'], inplace=True)
    
    # TODO change path to live cam path
    # Initialize the video capture object
    cap = cv2.VideoCapture("/dev/video1")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)



    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        exit()
    
    next_frame_id = 0
    counter = 0

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame is read correctly
        if not ret:
            print("Error: Unable to capture frame.")
            break

        _ , ids_cap_frame = detect_QR(frame)  # Detect QR codes in the frame
        if ids_cap_frame is not None:
            next_frame_id, next_qr = calculate_next_frame(frame,df)
            if next_frame_id == None:
                continue

            df = df[df.index.get_level_values('Frame ID') >= next_frame_id]
            next_frame_rows = get_rows_by_frame_id(df, next_frame_id)
            next_frame_data = next_frame_rows[next_frame_rows.index.get_level_values('QR ID') == next_qr[0]]
            
            
            command = get_movement_commands_for_this_frame(frame, next_frame_data, camera_matrix, dist_coeffs)
            counter+=1
            if not command == '' and counter % 10 == 0:
                print(command)
            
        cv2.imshow('Live Stream', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Error: No video file path provided.")
        print("Usage: python part_2.py <path_to_csv>")
        sys.exit(1)
    file_path = sys.argv[1]
    controller(file_path)
