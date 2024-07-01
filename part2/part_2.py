import numpy as np
import pandas as pd
import cv2



def best_single_command(translation, rotation):
    """
    Determines the single best movement command to bring the first ArUco marker closest to the second.
    
    Parameters:
    translation (numpy.ndarray): Translation difference between the two markers.
    rotation (numpy.ndarray): Rotation difference between the two markers.
    
    Returns:
    str: The best movement command.
    """
    commands = {
        "forward": translation[2],
        "backward": -translation[2],
        "right": translation[0],
        "left": -translation[0],
        "up": translation[1],
        "down": -translation[1],
        "turn right": rotation[2],
        "turn left": -rotation[2]
    }
    
    # Select the command with the maximum absolute value
    best_command = max(commands, key=commands.get)
    
    return best_command

def compute_transform(tvec1, rvec1, tvec2, rvec2):
    """
    Computes the translation and rotation needed to move from the first ArUco marker's position 
    and orientation to the second.
    
    Parameters:
    tvec1 (numpy.ndarray): Translation vector of the first ArUco marker.
    rvec1 (numpy.ndarray): Rotation vector of the first ArUco marker.
    tvec2 (numpy.ndarray): Translation vector of the second ArUco marker.
    rvec2 (numpy.ndarray): Rotation vector of the second ArUco marker.
    
    Returns:
    tuple: Translation difference and rotation difference between the two markers.
    """
    # Calculate translation difference
    translation = tvec2 - tvec1
    # Convert rotation vectors to rotation matrices
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    # Calculate rotation difference
    R_diff = R2 @ R1.T
    rvec_diff, _ = cv2.Rodrigues(R_diff)
    return translation, rvec_diff

def detect_aruco(frame, camera_matrix, dist_coeffs):
    """
    Detects the ArUco marker in the given frame and returns its translation and rotation vectors.
    
    Parameters:
    frame (numpy.ndarray): The image frame containing the ArUco marker.
    
    Returns:
    tuple: Translation vector (tvec) and rotation vector (rvec) of the detected ArUco marker.
    """

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        # Detect markers using ArUco dictionary
        corners, ids, _ = cv2.aruco.detectMarkers(gray_frame, cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100))

        if ids is not None and len(ids) > 0:
            # Estimate pose of the first detected marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
            return tvecs[0][0], rvecs[0][0]
        else:
            raise ValueError("No ArUco marker detected")

    except Exception as e:
        print(f"Error detecting markers: {e}")
        return None, None 

def get_best_movement_command(frame, next_frame_data, camera_matrix, dist_coeffs):
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
    tvec1, rvec1 = detect_aruco(frame, camera_matrix, dist_coeffs)
    tvec2, rvec2 = calculate_rvec_tvec(next_frame_data["distance"], next_frame_data["yaw"], next_frame_data["pitch"], next_frame_data["roll"])
    
    # Compute translation and rotation differences
    translation, rotation = compute_transform(tvec1, rvec1, tvec2, rvec2)
    
    # Generate the best single movement command based on the differences
    command = best_single_command(translation, rotation)
    
    return command


def isOverlap():
    pass


def eulerAnglesToRotationMatrix(yaw, pitch, roll):
    # Convert angles to radians
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    # Rotation matrix for yaw
    Rz_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Rotation matrix for pitch
    Ry_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # Rotation matrix for roll
    Rx_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Combined rotation matrix
    R = Rz_yaw @ Ry_pitch @ Rx_roll
    return R

def calculate_rvec_tvec(distance, yaw, pitch, roll):
    # Get the rotation matrix
    R = eulerAnglesToRotationMatrix(yaw, pitch, roll)

    # Convert rotation matrix to rotation vector
    rvec, _ = cv2.Rodrigues(R)

    # Create the translation vector (assuming distance is along the z-axis)
    tvec = np.array([[0], [0], [distance]])

    return tvec, rvec

def find_dest_qr_id(ids_frame, next_qr_ids):
    pass

def find_common_qr_id(ids_frame, next_qr_ids):
    # Convert lists to sets
    set1 = set(ids_frame)
    set2 = set(next_qr_ids)

    # Find the intersection of the sets
    common_elements = set1.intersection(set2)

    # Return one common element or None if no common elements
    if common_elements:
        return next(iter(common_elements))
    else:
        return None

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
    matching_rows = df[df['Frame ID'] == frame_id]
    if not matching_rows.empty:
        return matching_rows  # Return all rows that match the frame ID
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no matches are found

def get_next_frame_id(df, frame_id):

    next_frame_id = frame_id + 1
    max = df['Frame ID'].max()

    # Iterate until we find the next available frame ID or exceed the maximum frame ID in df
    while next_frame_id <= max:
        if (df['Frame ID'] == next_frame_id).any():
            return next_frame_id
        else:
            next_frame_id += 1
    
    return None


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

def calculate_next_frame(online_frame,df:pd.DataFrame):
    online_qrs = detect_QR(online_frame)[1]
    online_qrs = [x[0] for x in online_qrs]
    # Filter DataFrame to include only rows with QR IDs in the tuple
    df_filtered = df[df['QR ID'].isin(online_qrs)]
    max_frame_id_row = df_filtered.loc[df_filtered['Frame ID'].idxmax()]
    max_frame_id = max_frame_id_row['Frame ID']
    corresponding_qr_id = max_frame_id_row['QR ID']
    return max_frame_id, corresponding_qr_id
    

    
    

def controler(file_path):

    camera_matrix, dist_coeffs = initialize_camera_parameters()

    # Load the CSV file
    df = pd.read_csv(file_path, sep=None, engine='python')

    # Initialize the video capture object
    # cap = cv2.VideoCapture('/dev/video3')
    cap = cv2.VideoCapture('/home/oriaz/Desktop/QR_Recognitation/videos/online_sim.mp4')


    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        exit()
    # first_iteration = True
    next_frame_id = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame is read correctly
        if not ret:
            print("Error: Unable to capture frame.")
            break

        corners_cap_frame, ids_cap_frame = detect_QR(frame)  # Detect QR codes in the frame
        if ids_cap_frame is not None:
            # if first_iteration:
            # first_iteration = False
            next_frame_id, next_qr = calculate_next_frame(frame,df)
            df = df[df['Frame ID'] >= next_frame_id]
            # # find a target frame row in the csv
            # if not first_iteration:
            #     next_frame_id = get_next_frame_id(df, next_frame_id)
            next_frame_rows = get_rows_by_frame_id(df, next_frame_id)
            # get the corners of the cap frame
        
            index =  list(ids_cap_frame).index([next_qr]) 
            rvec1, tvec1, _ = cv2.aruco.estimatePoseSingleMarkers(corners_cap_frame[index], 0.05, camera_matrix, dist_coeffs)

            # get the corners of the csv frame
            next_frame_data = next_frame_rows[next_frame_rows['QR ID'] == 5]
            tvec2, rvec2 = calculate_rvec_tvec(next_frame_data["Distance"], 
                                             next_frame_data["Yaw (degrees)"], 
                                             next_frame_data["Pitch (degrees)"], 
                                             next_frame_data["Roll (degrees)"])

            while(isOverlap(rvec1, tvec1, tvec2, rvec2 ) == False):
                print(get_best_movement_command(frame, next_frame_data, camera_matrix, dist_coeffs))

        # Display the frame
        cv2.imshow('Live Stream', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    file_path = '../outputs/challengeB_data.csv'
    controler(file_path)
# if __name__ == "__main__":
#     df = pd.read_csv('../outputs/challengeB_data.csv', sep=None, engine='python')
#     filtered_df = df[(df['Frame ID'] >= 425) & (df['Frame ID'] <= 427)]
#     calculate_first_frame('banana',filtered_df)
