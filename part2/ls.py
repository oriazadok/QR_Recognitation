import cv2

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

# Loop to continuously capture frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame is read correctly
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Display the frame
    cv2.imshow('Live Stream', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()

