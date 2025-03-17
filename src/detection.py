import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from motion_detection import MotionDetector
from action_recognition import ActionRecognizer

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Khởi tạo Motion & Action detection
motion_detector = MotionDetector()
action_recognizer = ActionRecognizer()

def detect_head_pose(frame):
    """
    Phát hiện hướng quay đầu dựa trên vector mắt-mũi.
    """
    results = yolo_model(frame)
    cheating_detected = False

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Cắt ảnh người để phân tích tư thế
            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size > 0:
                person_roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                results_pose = pose.process(person_roi_rgb)

                if results_pose.pose_landmarks:
                    landmarks = results_pose.pose_landmarks.landmark

                    # Lấy tọa độ mắt & mũi
                    nose = np.array([landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y])
                    left_eye = np.array([landmarks[mp_pose.PoseLandmark.LEFT_EYE].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE].y])
                    right_eye = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_EYE].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE].y])

                    # Vector hướng mũi
                    eye_center = (left_eye + right_eye) / 2
                    nose_vector = nose - eye_center

                    # Ngưỡng xác định hướng quay đầu
                    angle_x = np.arctan2(nose_vector[0], nose_vector[1]) * 180 / np.pi

                    direction = "Looking Straight"
                    if angle_x > 15:
                        direction = "Looking Left"
                        cheating_detected = True
                    elif angle_x < -15:
                        direction = "Looking Right"
                        cheating_detected = True

                    # Hiển thị hướng quay đầu
                    cv2.putText(frame, direction, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return frame, cheating_detected
