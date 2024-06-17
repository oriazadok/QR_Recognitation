import sys
import cv2
import pyzbar.pyzbar as pyzbar

def extract_frames(video_path):
    video_capture = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)
    video_capture.release()
    return frames

def detect_qr_codes(frame):

    print(frame)

    # Convert to RGB (if necessary)
    if cv2.cvtColor is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Decode QR codes
    qr_codes = pyzbar.decode(frame)

    # Extract decoded data
    data_strings = []
    for code in qr_codes:
        data_strings.append(code.data.decode("utf-8"))

    print(data_strings)

    return data_strings
    # qr_detector = cv2.QRCodeDetector()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # data, points, _ = qr_detector.detectAndDecode(gray)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect QR code
    qr_detector = cv2.QRCodeDetector()
    data, points, _ = qr_detector.detectAndDecode(gray)
    # data, points, _ = qr_detector.detectAndDecode(frame)
    print("pointssss: ", points)
    if points is not None:
        points = points.astype(int)
    return points

def draw_rectangles(frame, points):
    if points is not None and len(points) > 0:
        points = points[0]  # Assuming single QR code for simplicity
        for i in range(len(points)):
            pt1 = tuple(points[i])
            pt2 = tuple(points[(i + 1) % len(points)])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 5)
    return frame

def reassemble_video(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()

def process_video(input_path, output_path):
    frames = extract_frames(input_path)
    processed_frames = []
    for frame in frames:
        points = detect_qr_codes(frame)
        processed_frame = draw_rectangles(frame, points)
        processed_frames.append(processed_frame)
    reassemble_video(processed_frames, output_path)

if __name__ == "__main__":
    input_video_path = sys.argv[1]
    output_video_path = 'outputs/out.mp4'
    process_video(input_video_path, output_video_path)
