import sys

import cv2
import numpy as np
import pandas as pd
import os
import shutil

# Camera parameters
resolution = (1280, 720)  # 720p resolution
field_of_view = 82.6  # Field of View in degrees

# Compute the focal length (assuming square pixels)
focal_length = (resolution[0] / 2) / np.tan(np.deg2rad(field_of_view / 2))
camera_matrix = np.array([[focal_length, 0, resolution[0] / 2],
                          [0, focal_length, resolution[1] / 2],
                          [0, 0, 1]])
distortion_coeffs = np.zeros((4, 1))

# Function to prepare directories
def prepare_directories(dirs):
    for dir in dirs:
        if os.path.exists(dir):
            # Remove the directory and its contents
            shutil.rmtree(dir)
        os.makedirs(dir)

# Function to compute 3D pose information
def compute_3d_pose(rvec, tvec):
    # Distance calculation
    distance = np.linalg.norm(tvec)

    # Yaw, pitch, and roll calculations
    R, _ = cv2.Rodrigues(rvec)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = np.arctan2(R[2, 1], R[2, 2])

    return distance, yaw, pitch, roll

# Ensure OpenCV includes the aruco module
if not hasattr(cv2, 'aruco'):
    raise ImportError("OpenCV is not compiled with aruco module. Please install OpenCV with aruco support.")

# Load Aruco dictionary and set parameters
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
detector_params = cv2.aruco.DetectorParameters_create()

# Function to process a video
def process_video_file(video_path, csv_output, video_output, frames_output_dir):
    cap = cv2.VideoCapture(video_path)

    # Check if video capture is successful
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Get video properties
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0:
        print(f"Error: Frame rate of the video file {video_path} is zero.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay = int(1000 / frame_rate) if frame_rate > 0 else 1  # Avoid division by zero

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output, fourcc, frame_rate, (width, height))

    # Data for CSV output
    csv_data = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        try:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray_frame, aruco_dict, parameters=detector_params)
        except Exception as e:
            print(f"Error detecting markers: {e}")
            break

        if ids is not None:
            for i in range(len(ids)):
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.05, camera_matrix, distortion_coeffs)

                # Draw marker and axis
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.drawFrameAxes(frame, camera_matrix, distortion_coeffs, rvec, tvec, 0.1)

                # Get 2D corner points
                corner_points = corners[i][0]
                corner_points_list = corner_points.tolist()

                # Draw rectangle around marker
                pts = np.array(corner_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {ids[i][0]}", (int(corner_points[0][0]), int(corner_points[0][1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                # Compute 3D pose
                distance, yaw, pitch, roll = compute_3d_pose(rvec, tvec)

                # Convert angles to degrees
                yaw, pitch, roll = np.degrees([yaw, pitch, roll])

                # Append data to CSV list
                csv_data.append([frame_count, ids[i][0], corner_points_list, distance, yaw, pitch, roll])

            # Save the current frame
            frame_file = os.path.join(frames_output_dir, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_file, frame)

        # Write frame to output video
        out.write(frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

        frame_count += 1

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save CSV data
    columns = ['Frame ID', 'QR ID', 'QR 2D (Corner Points)', 'Distance', 'Yaw (degrees)', 'Pitch (degrees)',
               'Roll (degrees)']
    df = pd.DataFrame(csv_data, columns=columns)
    df.to_csv(csv_output, index=False)



if __name__ == "__main__":
    input_video_path = sys.argv[1]
    csv_file = 'output/output_data.csv'
    frames_output_directory = 'output/img'
    output_video_path = 'outputs/out.mp4'
    process_video_file(input_video_path, csv_file, output_video_path, frames_output_directory)
