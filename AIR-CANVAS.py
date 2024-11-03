#importing libraries
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize color points
bpoints, gpoints, rpoints, ypoints = [deque(maxlen=1024) for _ in range(4)]
blue_index = green_index = red_index = yellow_index = 0

# Define colors and set up paint window
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Create paint window
paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255
buttons = [(40, "CLEAR"), (160, "BLUE"), (275, "GREEN"), (390, "RED"), (505, "YELLOW")]

for (x, label), color in zip(buttons, colors + [(0, 0, 0)]):
    cv2.rectangle(paintWindow, (x, 1), (x + 100, 65), color, 2)
    cv2.putText(paintWindow, label, (x + 5, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    # Draw buttons on the frame
    for (x, label), color in zip(buttons, colors + [(0, 0, 0)]):
        cv2.rectangle(frame, (x, 1), (x + 100, 65), color, 2)
        cv2.putText(frame, label, (x + 5, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Process hand landmarks if detected
    if result.multi_hand_landmarks:
        landmarks = []
        for hand_landmark in result.multi_hand_landmarks:
            for lm in hand_landmark.landmark:
                lmx, lmy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, hand_landmark, mpHands.HAND_CONNECTIONS)

        # Track fingertip and thumb for drawing
        fore_finger, thumb = tuple(landmarks[8]), tuple(landmarks[4])
        cv2.circle(frame, fore_finger, 3, (0, 255, 0), -1)

        # Clear points if thumb and index are close
        if abs(thumb[1] - fore_finger[1]) < 30:
            bpoints, gpoints, rpoints, ypoints = [[deque(maxlen=512)] for _ in range(4)]
            blue_index = green_index = red_index = yellow_index = 0
            paintWindow[67:, :, :] = 255

        # Handle color selection at the top bar
        elif fore_finger[1] <= 65:
            if 40 <= fore_finger[0] <= 140:   # Clear
                bpoints, gpoints, rpoints, ypoints = [[deque(maxlen=512)] for _ in range(4)]
                blue_index = green_index = red_index = yellow_index = 0
                paintWindow[67:, :, :] = 255
            elif 160 <= fore_finger[0] <= 255:  # Blue
                colorIndex = 0
            elif 275 <= fore_finger[0] <= 370:  # Green
                colorIndex = 1
            elif 390 <= fore_finger[0] <= 485:  # Red
                colorIndex = 2
            elif 505 <= fore_finger[0] <= 600:  # Yellow
                colorIndex = 3
        else:
            # Add points for the selected color
            points = [bpoints, gpoints, rpoints, ypoints][colorIndex]
            points[blue_index].appendleft(fore_finger)

    # Draw lines on frame and paintWindow
    points = [bpoints, gpoints, rpoints, ypoints]
    for color_pts, color in zip(points, colors):
        for pt in color_pts:
            for k in range(1, len(pt)):
                if pt[k - 1] is None or pt[k] is None:
                    continue
                cv2.line(frame, pt[k - 1], pt[k], color, 2)
                cv2.line(paintWindow, pt[k - 1], pt[k], color, 2)

    # Display output frames
    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
