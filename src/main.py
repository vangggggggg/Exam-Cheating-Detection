import cv2
from detection import detect_head_pose
from motion_detection import MotionDetector
from action_recognition import ActionRecognizer

# Khởi tạo các mô-đun
motion_detector = MotionDetector()
action_recognizer = ActionRecognizer()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Phát hiện tư thế đầu
    frame, head_pose_cheating = detect_head_pose(frame)

    # Phát hiện chuyển động
    frame, motion_detected = motion_detector.detect_motion(frame)

    # Phát hiện hành động gian lận
    frame, action_detected = action_recognizer.detect_cheating_action(frame)

    # Nếu phát hiện gian lận
    if head_pose_cheating or motion_detected or action_detected:
        cv2.putText(frame, "ALERT: Cheating Detected!", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow('Exam Surveillance', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
