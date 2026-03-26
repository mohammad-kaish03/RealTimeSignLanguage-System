import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
from config import DATA_PATH

# Ensure data folder exists
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

# 10 Signs Mapping
SIGNS = {
    "1": "HELLO",
    "2": "YES",
    "3": "NO",
    "4": "THANKS",
    "5": "PLEASE",
    "6": "HELP",
    "7": "STOP",
    "8": "OK",
    "9": "YOU",
    "0": "ME"
}


# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# Webcam Setup
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access camera")
    exit()

current_label = None
rows = []

print("\n✅ SIGN LANGUAGE DATA COLLECTION STARTED")
print("---------------------------------------")
print("Select label by pressing keys:")
for k, v in SIGNS.items():
    print(f"  {k} => {v}")

print("\nControls:")
print("  s  => Save sample")
print("  q  => Quit and save dataset\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera not working!")
        break

    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    landmarks = None

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract 21 landmarks (x, y, z) => 63 features
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        cv2.putText(frame, f"Label: {current_label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "No hand detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, "Press s=save | q=quit", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Data Collection - Sign Language", frame)

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):
        break

    # Save sample
    if key == ord("s"):
        if landmarks is not None and current_label is not None:
            rows.append(landmarks + [current_label])
            print(f"✅ Saved: {current_label} | Total samples: {len(rows)}")
        else:
            print("❌ Cannot save: No hand detected OR label not selected")

    # Select label (1,2,3...)
    try:
        pressed = chr(key)
        if pressed in SIGNS:
            current_label = SIGNS[pressed]
            print("🎯 Selected label:", current_label)
    except:
        pass

cap.release()
cv2.destroyAllWindows()


# Save dataset.csv
if len(rows) > 0:
    cols = [f"f{i}" for i in range(63)] + ["label"]
    new_df = pd.DataFrame(rows, columns=cols)

    # Append if already exists
    if os.path.exists(DATA_PATH):
        old_df = pd.read_csv(DATA_PATH)
        final_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    final_df.to_csv(DATA_PATH, index=False)
    print(f"\n✅ Dataset saved successfully at: {DATA_PATH}")
    print("✅ Total samples in file:", len(final_df))
else:
    print("\n⚠️ No samples collected. Dataset not saved.")
