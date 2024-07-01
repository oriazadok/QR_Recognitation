# import cv2
# import logging
# import time

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def open_stream(url):
#     logging.info(f"Attempting to open stream at {url}")
#     return cv2.VideoCapture(url)

# def main():
#     # Change your IP
#     url = 'https://172.20.10.7:8080'
#     cap = open_stream(url)
    
#     while True:
#         if not cap.isOpened():
#             logging.error("Failed to open stream.")
#             time.sleep(5)  # wait before retrying
#             cap = open_stream(url)
#             continue

#         ret, frame = cap.read()
        
#         if not ret:
#             logging.warning("Stream ended or frame not captured.")
#             cap.release()
#             time.sleep(5)  # wait before retrying
#             cap = open_stream(url)
#             continue
        
#         try:
#             cv2.imshow('temp', cv2.resize(frame, (600, 400)))

#             key = cv2.waitKey(1)
#             if key == ord('q'):
#                 break
#         except cv2.error as e:
#             logging.error(f"OpenCV error: {e}")
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()




import cv2
import logging
import os


def main():
    device_path = '/dev/video3'
    cap = cv2.VideoCapture(device_path)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open video stream from webcam")
        exit()

    # Capture video feed
    while True:
        ret, frame = cap.read()
           
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Display the frame
        cv2.imshow('Webcam Feed', frame)

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
