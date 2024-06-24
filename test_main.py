import pytest
import cv2
from main import detect_QR, initialize_camera_parameters, compute_3d_pose


class TestQRCodeDetection:
    
    @staticmethod
    def read_frame(path):
        frame = cv2.imread(path)
        return detect_QR(frame)
    
    def test_blurry_frame(self):
        corners, ids = self.read_frame('TestData/frames/blurry_image_frame.jpg')
        assert ids is None
    
    def test_cutted_frame(self):
        corners, ids = self.read_frame('TestData/frames/cutted_image_frame.jpg')
        assert ids is None

    def test_no_QR_frame(self):
        corners,ids = self.read_frame('TestData/frames/no_qr_frame.jpg')
        assert ids == None

    def test_basic_frame(self):
        corners, ids = self.read_frame('TestData/frames/basic_case_frame.jpg')
        assert ids is not None
        assert ids[0] == [1]
        assert (corners[0][0][0] == [712., 116.]).all()

    def test_2_frame(self):
        corners,ids = self.read_frame('TestData/frames/2_qr_frame.jpg')
        assert len(ids) == 2
        assert (corners[0][0][0] == [69., 77.]).all()
        assert (corners[1][0][0] == [604.,  61.]).all()

    
class TestCompute3DPose:

    @classmethod
    def setup_class(cls):
        cls.camera_matrix, cls.distortion_coeffs = initialize_camera_parameters()


    @staticmethod
    def read_frame(path):
        frame = cv2.imread(path)
        return detect_QR(frame)
    
    def test_compute_3d_pose(self):
        corners, ids = self.read_frame('TestData/frames/basic_case_frame.jpg')
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], 0.05, self.camera_matrix, self.distortion_coeffs)
        distance, yaw, pitch, roll = compute_3d_pose(rvec, tvec)
        print(distance, yaw, pitch, roll)
        assert (distance, yaw, pitch, roll) == (1.3624498135056085, 0.14899773008818532, 0.18643957378132553, -3.067477730483944)

    def test_compute_3d_pose(self):
        corners, ids = self.read_frame('TestData/frames/2_qr_frame.jpg')

        for i in range(len(ids)):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.05, self.camera_matrix, self.distortion_coeffs)
            distance, yaw, pitch, roll = compute_3d_pose(rvec, tvec)
            print(distance, yaw, pitch, roll)
            if i == 0:
                assert (distance, yaw, pitch, roll) == (2.72012630310793, 0.046164401138392375, -0.02806027493828094, -3.079590743916259)
            else:
                assert (distance, yaw, pitch, roll) == (0.9921962876750312, 0.1834169802071622, 0.6520330514984353, -3.013564602120967)