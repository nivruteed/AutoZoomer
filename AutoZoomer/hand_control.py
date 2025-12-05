import cv2
import mediapipe as mp
import pyautogui
import math
import time

# ===== MediaPipe Setup =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ===== Webcam Setup =====
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

scroll_prev_y = None
clicking = False
screenshot_cooldown = 1  # seconds
last_screenshot_time = 0

def get_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

        h, w, c = img.shape
        lm = hand.landmark

        # Finger Tips
        index_x, index_y = int(lm[8].x * w), int(lm[8].y * h)
        thumb_x, thumb_y = int(lm[4].x * w), int(lm[4].y * h)
        middle_x, middle_y = int(lm[12].x * w), int(lm[12].y * h)

        # ===== Feature 1: Scroll (Swipe Up/Down) =====
        if scroll_prev_y is not None:
            diff_scroll = scroll_prev_y - index_y
            if diff_scroll > 25:     # Swipe Up → scroll down
                pyautogui.scroll(-300)
            elif diff_scroll < -25:  # Swipe Down → scroll up
                pyautogui.scroll(300)
        scroll_prev_y = index_y

        # ===== Feature 2: Air Mouse =====
        pyautogui.moveTo(index_x, index_y, duration=0.05)

        # Left click: Thumb + Index pinch
        pinch_dist = get_distance(index_x, index_y, thumb_x, thumb_y)
        if pinch_dist < 25 and not clicking:
            pyautogui.click()
            clicking = True
        if pinch_dist >= 25:
            clicking = False

        # Right click: Thumb + Middle pinch
        right_dist = get_distance(thumb_x, thumb_y, middle_x, middle_y)
        if right_dist < 25:
            pyautogui.rightClick()

        # ===== Feature 3: Gesture Screenshot (“OK” gesture) =====
        circle_dist = get_distance(thumb_x, thumb_y, index_x, index_y)
        middle_up = lm[12].y < lm[9].y  # Middle finger up
        current_time = time.time()
        if circle_dist < 20 and middle_up and (current_time - last_screenshot_time > screenshot_cooldown):
            pyautogui.screenshot("gesture_capture.png")
            last_screenshot_time = current_time
            cv2.putText(img, "Screenshot Taken!", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Hand Control System", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
