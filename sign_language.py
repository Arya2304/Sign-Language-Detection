import tkinter as tk
from tkinter import messagebox
import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class SignLanguageDetectorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Sign Language Detector")
        self.master.geometry("300x200")

        self.label = tk.Label(master, text="Welcome to Sign Language Detector!")
        self.label.pack()

        self.question_label = tk.Label(master, text="Do you want to start the sign detection session?")
        self.question_label.pack()

        self.yes_button = tk.Button(master, text="Yes", command=self.start_session)
        self.yes_button.pack(side=tk.LEFT)

        self.no_button = tk.Button(master, text="No", command=self.master.destroy)
        self.no_button.pack(side=tk.RIGHT)

    def start_session(self):
        self.master.withdraw()
        self.instruction_window = tk.Toplevel(self.master)
        self.instruction_window.title("Instructions")
        self.instruction_window.geometry("300x200")

        self.instruction_label = tk.Label(self.instruction_window, text="Please place your hand in front of the camera.")
        self.instruction_label.pack()

        self.instruction_label2 = tk.Label(self.instruction_window, text="Make sure your hand is visible and well-lit.")
        self.instruction_label2.pack()

        self.start_button = tk.Button(self.instruction_window, text="Start", command=self.run_detection)
        self.start_button.pack()

        self.quit_button = tk.Button(self.instruction_window, text="Quit", command=self.master.destroy)
        self.quit_button.pack()

    def run_detection(self):
        try:
            model_dict = pickle.load(open('./model.p', 'rb'))
            model = model_dict['model']
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        # Define the labels dictionary
        labels_dict = {0: 'A', 1: 'B', 2: 'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'}

        # Set up the MediaPipe hands module with static_image_mode=False for video
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

        # Set up the camera
        cap = cv2.VideoCapture(0)

        while True:
            data_aux = []
            x_ = []
            y_ = []

            ret, frame = cap.read()

            if not ret:
                print("Failed to read frame. Exiting...")
                break

            H, W, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        mp_hands.HAND_CONNECTIONS,  # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                x1 = max(int(min(x_) * W) - 10, 0)
                y1 = max(int(min(y_) * H) - 10, 0)

                x2 = max(int(max(x_) * W) - 10, 0)
                y2 = max(int(max(y_) * H) - 10, 0)

                prediction = model.predict([np.asarray(data_aux)])

                predicted_character = labels_dict[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)
                print(predicted_character , end="")
            cv2.imshow('Hand Detection Started!', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

root = tk.Tk()
my_gui = SignLanguageDetectorGUI(root)
root.mainloop()
