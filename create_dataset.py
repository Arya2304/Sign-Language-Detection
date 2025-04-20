import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        try:
            # Read the image
            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            if img is None:
                print(f"Error: Failed to read image {img_path}")
                continue

            # Convert the image to RGB format
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process the image to detect hands
            results = hands.process(img_rgb)

            # Extract hand landmark data if hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux = []
                    min_x = min(landmark.x for landmark in hand_landmarks.landmark)
                    min_y = min(landmark.y for landmark in hand_landmarks.landmark)
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min_x)  # Normalize x-coordinate
                        data_aux.append(y - min_y)  # Normalize y-coordinate
                    data.append(data_aux)
                    labels.append(dir_)  # Use dir_ as the label

                    # Optional: Visualize the image with landmarks (if any)
                    mp_drawing.draw_landmarks(img_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                                mp_drawing_styles.get_default_hand_connections_style())
                    plt.imshow(img_rgb)
            else:
                print(f"No hands detected in image {img_path}")
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

# Validate data and labels
if len(data) != len(labels):
    print("Error: Data and labels are not of the same length")
else:
    try:
        # Save data to pickle file
        f = open('data.pickle', 'wb')
        pickle.dump({'data': data, 'labels': labels}, f)
        f.close()
        print("Data processing complete!")
    except Exception as e:
        print(f"Error saving data to pickle file: {e}")

plt.show()  # Display any previously plotted images