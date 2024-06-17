import pytest
import cv2
from PIL import Image, ImageTk
import numpy as np

import os
from main import detect_qr_codes, draw_rectangles

# @pytest.fixture
# def test_frame():
#     # Function to create a QR code image
#     def create_qr_code(data, size=200):
#         qr = qrcode.QRCode(version=1, box_size=10, border=5)
#         qr.add_data(data)
#         qr.make(fit=True)
#         img = qr.make_image(fill='black', back_color='white')
#         img = img.resize((size, size))
#         return np.array(img.convert('RGB'))

#     # Function to create a test frame with a QR code
#     def create_test_frame(data, frame_size=(640, 480)):
#         frame = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255
#         qr_code_img = create_qr_code(data)
#         x_offset = (frame_size[0] - qr_code_img.shape[1]) // 2
#         y_offset = (frame_size[1] - qr_code_img.shape[0]) // 2
#         frame[y_offset:y_offset+qr_code_img.shape[0], x_offset:x_offset+qr_code_img.shape[1]] = qr_code_img
#         return frame

#     # Generate a test frame
#     frame = create_test_frame('Test QR Code')
#     return frame

def test_detect_qr_code():
    # frame1 = Image.open('TestData/frames/frames/frame_282.jpg')
    video_capture = cv2.VideoCapture('TestData/frames/frames/frame_3.jpg')
    ret, frame = video_capture.read()
    
    points = detect_qr_codes(frame)
    print(points)
    draw_rectangles(frame,points)
    output_dir = 'TestData/TestsOutput'
    output_path = os.path.join(output_dir, 'test_frame_3.jpg')

    if ret:
        # Save the frame as an image
        cv2.imwrite('TestData/TestsOutput/output_image.jpg', frame)
        print("Image saved successfully.")
    else:
        print("Error: Unable to read the frame from the video capture.")
    
    # Ensure the output directory exists
    # os.makedirs(output_dir, exist_ok=True)
    # frame.save(output_path)

test_detect_qr_code()



# def test_detect_qr_codes(test_frame):
#     points = detect_qr_codes(test_frame)
#     assert len(points) > 0, "QR code not detected"
#     assert len(points[0]) == 4, "QR code corners not detected correctly"
#     for point in points[0]:
#         assert len(point) == 2, "Each point should have x and y coordinates"

