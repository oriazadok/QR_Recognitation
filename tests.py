import pytest
import cv2
from PIL import Image, ImageTk
import numpy as np

import os
from main import detect_QR

# TODO make proper pytest tests

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


def read_frame(path):
    frame = cv2.imread(path)
    return detect_QR(frame)
    

def test_blurry_frame():
    corners,ids = read_frame('TestData/frames/blurry_image_frame.jpg')
    assert ids == None
    
def test_cutted_frame():
    corners,ids = read_frame('TestData/frames/cutted_image_frame.jpg')
    assert ids == None

def test_basic_frame():
    corners,ids = read_frame('TestData/frames/basic_case_frame.jpg')
    assert ids[0] == [1]
    print(corners)
    assert (corners[0][0][0] == [712., 116.]).all()
    

test_basic_frame()
test_cutted_frame()
test_blurry_frame()


# TODO compute_3d_pose test by mocking the input values and assert equals to excpected output




