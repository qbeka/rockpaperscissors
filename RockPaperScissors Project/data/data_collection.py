import cv2
import mediapipe as mp
import csv
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

GESTURES = ['rock', 'paper', 'scissors']

#Open webcam
cap = cv2.VideoCapture(0)

def save_landmark_data(label, landmarks):
    """
    Save the hand landmarks to a CSV file.
    Each row is labeled with the gesture (rock/paper/scissors).
    """
    with open('hand_landmarks.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        data = [label] + landmarks
        writer.writerow(data)

def collect_data_for_gesture(gesture_name):
    """
    Display the gesture name on the screen and collect hand landmark data.
    """
    print(f"Perform the gesture for {gesture_name.upper()}...")
    start_time = time.time()
    while time.time() - start_time < 15: 
        success, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                landmarks = []
                for lm in hand_lms.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)

                save_landmark_data(gesture_name, landmarks)
        
        cv2.putText(img, f'Collecting: {gesture_name.upper()}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Data Collection", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

while True:
    for gesture in GESTURES:
        collect_data_for_gesture(gesture)
    break

cap.release()
cv2.destroyAllWindows()
print("Data collection complete! Data saved to 'hand_landmarks.csv'.")
