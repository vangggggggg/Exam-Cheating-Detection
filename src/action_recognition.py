import mediapipe as mp
import cv2

mp_holistic = mp.solutions.holistic

class ActionRecognizer:
    def __init__(self):
        self.holistic = mp_holistic.Holistic()

    def detect_cheating_action(self, frame):
        """
        Phát hiện hành động gian lận (che mắt, nhìn sang bên, đưa tay lên mặt).
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)

        cheating_detected = False

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_hand = landmarks[mp_holistic.PoseLandmark.LEFT_WRIST]
            right_hand = landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST]
            nose = landmarks[mp_holistic.PoseLandmark.NOSE]

            # Kiểm tra nếu tay gần mặt (che mắt, dùng điện thoại)
            if abs(left_hand.y - nose.y) < 0.15 or abs(right_hand.y - nose.y) < 0.15:
                cv2.putText(frame, "Cheating Action Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cheating_detected = True

        return frame, cheating_detected
