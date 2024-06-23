import sys
import cv2
import numpy as np
import pandas as pd
import os
import shutil

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

# Function to prepare directories for output, clearing existing ones
def prepare_directories(dirs):
    for dir in dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)  # Remove existing directory
        os.makedirs(dir)  # Create a new directory

# Function to compute the 3D pose (distance, yaw, pitch, roll) from rotation and translation vectors
def compute_3d_pose(rvec, tvec):
    distance = np.linalg.norm(tvec)  # Calculate the distance
    R, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to rotation matrix
    yaw = np.arctan2(R[1, 0], R[0, 0])  # Calculate yaw angle
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))  # Calculate pitch angle
    roll = np.arctan2(R[2, 1], R[2, 2])  # Calculate roll angle

    return distance, yaw, pitch, roll

# Function to detect QR codes in a given frame
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
    
# Function to write data to CSV list and annotate the frame with detected QR codes
def w2csvList(i, corners, frame, ids, rvec, tvec, frame_count):
    corner_points = corners[i][0]  # Get corner points of the QR code
    corner_points_list = corner_points.tolist()  # Convert to list

    # Draw polylines around the QR code
    pts = np.array(corner_points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
    # Put QR code ID text on the frame
    cv2.putText(frame, f"ID: {ids[i][0]}", (int(corner_points[0][0]), int(corner_points[0][1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Compute 3D pose (distance, yaw, pitch, roll) of the QR code
    distance, yaw, pitch, roll = compute_3d_pose(rvec, tvec)
    yaw, pitch, roll = np.degrees([yaw, pitch, roll])  # Convert to degrees

    # Append data to CSV list
    csv_data.append([frame_count, ids[i][0], corner_points_list, distance, yaw, pitch, roll])

# Function to write the collected QR code data to a CSV file
def write2csv(csv_output):
    columns = ['Frame ID', 'QR ID', 'QR 2D (Corner Points)', 'Distance', 'Yaw (degrees)', 'Pitch (degrees)',
               'Roll (degrees)']
    df = pd.DataFrame(csv_data, columns=columns)  # Create a DataFrame from the data
    df.to_csv(csv_output, index=False)  # Write DataFrame to CSV file

# Function to write the collected QR code data to a CSV file
def write2csv(csv_output):
    columns = ['Frame ID', 'QR ID', 'QR 2D (Corner Points)', 'Distance', 'Yaw (degrees)', 'Pitch (degrees)',
               'Roll (degrees)']
    df = pd.DataFrame(csv_data, columns=columns)  # Create a DataFrame from the data
    df.to_csv(csv_output, index=False)  # Write DataFrame to CSV file


# Main function to process the video file
def process_video_file(video_path, csv_output, video_output, frames_output_dir, resolution=(1280, 720), field_of_view=82.6):
    camera_matrix, distortion_coeffs = initialize_camera_parameters(resolution, field_of_view)
    
    cap = cv2.VideoCapture(video_path)  # Open the video file
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    frame_rate = cap.get(cv2.CAP_PROP_FPS)   # Get the frame rate of the video
    if frame_rate == 0:
        print(f"Error: Frame rate of the video file {video_path} is zero.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get the width of the frames
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get the height of the frames
    delay = int(1000 / frame_rate) if frame_rate > 0 else 1  # Calculate delay between frames

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the coder decoder for output video
    out = cv2.VideoWriter(video_output, fourcc, frame_rate, (width, height))  # Create a VideoWriter object
    
    frame_count = 0
    while cap.isOpened():

        ret, frame = cap.read()  # Read a frame
        if not ret:
            break

        corners, ids = detect_QR(frame)  # Detect QR codes in the frame
        
        if ids is not None:
            for i in range(len(ids)):
                # Estimate the pose of the detected QR codes
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.05, camera_matrix, distortion_coeffs)
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)  # Draw the detected markers
                cv2.drawFrameAxes(frame, camera_matrix, distortion_coeffs, rvec, tvec, 0.1)  # Draw the pose axes

                w2csvList(i, corners, frame, ids, rvec, tvec, frame_count)  # Write data to CSV list and annotate frame

            frame_file = os.path.join(frames_output_dir, f'frame_{frame_count}.jpg')  # Define the frame file path
            cv2.imwrite(frame_file, frame)  # Save the annotated frame

        out.write(frame)  # Write the frame to the output video
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()  # Release the video capture object
    out.release()  # Release the video writer object
    cv2.destroyAllWindows()  # Close all OpenCV windows

    write2csv(csv_output)   # Write the QR's data into csv file


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Error: No video file path provided.")
        print("Usage: python main.py <path_to_video>")
        sys.exit(1)
    input_video_path = sys.argv[1]
    input_base_name = os.path.splitext(os.path.basename(input_video_path))[0]

    output_video_path = os.path.join('outputs', f'{input_base_name}_out.mp4')
    frames_output_directory = os.path.join('outputs', f'{input_base_name}_frames')
    csv_file = os.path.join('outputs', f'{input_base_name}_data.csv')

    # Global list to store CSV data
    csv_data = []

    # Prepare the output directories
    prepare_directories([frames_output_directory])

    # Process the video file
    process_video_file(input_video_path, csv_file, output_video_path, frames_output_directory)
