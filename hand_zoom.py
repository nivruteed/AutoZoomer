import cv2
import mediapipe as mp
import pyautogui
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_distance = 0
zoom_threshold = 30

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            x1 = handLms.landmark[4].x * img.shape[1]
            y1 = handLms.landmark[4].y * img.shape[0]

            x2 = handLms.landmark[8].x * img.shape[1]
            y2 = handLms.landmark[8].y * img.shape[0]

            distance = math.hypot(x2 - x1, y2 - y1)

            if prev_distance != 0:
                if distance - prev_distance > zoom_threshold:
                    pyautogui.hotkey('ctrl', '+')
                    print("Zoom In")

                elif prev_distance - distance > zoom_threshold:
                    pyautogui.hotkey('ctrl', '-')
                    print("Zoom Out")

            prev_distance = distance

    cv2.imshow("Hand Zoom Control", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
