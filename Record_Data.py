import cv2
import mediapipe as mp
import csv
import os

# --- SETUP ---
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
# Detect up to 2 hands with higher confidence for better stability
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Create CSV header if it doesn't exist
file_path = 'gestures.csv'
if not os.path.exists(file_path):
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        # 1 Label + 42 Left Hand Points + 42 Right Hand Points = 85 Columns
        header = ['label'] + [f'L{i}' for i in range(42)] + [f'R{i}' for i in range(42)]
        writer.writerow(header)

print("--- RECORDING CONTROLS ---")
print("1: Forward | 2: Backward | 3: Left | 4: Right")
print("5: F-Left  | 6: F-Right  | 7: B-Left | 8: B-Right")
print("Press ESC to Quit")

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1) # Flip so it acts like a mirror
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Initialize empty data for both hands (all zeros)
    left_hand_data = [0] * 42
    right_hand_data = [0] * 42

    if results.multi_hand_landmarks:
        # results.multi_handedness tells us if it's "Left" or "Right"
        for i, hand_handedness in enumerate(results.multi_handedness):
            label_side = hand_handedness.classification[0].label # "Left" or "Right"
            hand_lms = results.multi_hand_landmarks[i]
            
            # Extract coordinates
            pts = []
            for lm in hand_lms.landmark:
                pts.extend([lm.x, lm.y])
            
            # Assign to the correct slot
            if label_side == "Left":
                left_hand_data = pts
            else:
                right_hand_data = pts

            # Draw on screen
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    # inputs
    key = cv2.waitKey(1)
    action_label = None
    
    # labels
    mapping = {
        ord('1'): "Forward", ord('2'): "Backward", 
        ord('3'): "Left",    ord('4'): "Right",
        ord('5'): "F_Left",  ord('6'): "F_Right",
        ord('7'): "B_Left",  ord('8'): "B_Right"
    }
    
    if key in mapping:
        action_label = mapping[key]
        full_row = [action_label] + left_hand_data + right_hand_data
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(full_row)
        print(f"Recorded 1 frame of: {action_label}")

    cv2.imshow("Multi-Hand Data Collector", img)
    if key == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()