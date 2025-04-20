# Sign-Language-Detection

1. create_img.py:

This script captures images from the webcam to create a dataset for 26 classes (likely representing letters A-Z).
It saves 100 images per class in separate folders under the ./data directory.
The user is prompted to press "Q" to start capturing images for each class.

2. create_dataset.py:

This script processes the captured images in the ./data directory.
It uses MediaPipe to detect hand landmarks in each image.
The hand landmark coordinates are normalized and stored along with their labels (class folder names).
The processed data and labels are saved into a pickle file named data.pickle.
Optionally, it visualizes the hand landmarks on the images.

3. train_classifier.py:

This script loads the processed data from data.pickle.
It preprocesses the data by padding or truncating feature vectors to a uniform length.
A RandomForestClassifier is trained on the data.
The model is evaluated on a test split, and the accuracy is printed.
The trained model is saved to a pickle file named model.p.

4. inference_classifier.py:

This script loads the trained model from model.p.
It uses MediaPipe to detect hand landmarks in real-time from the webcam feed.
The landmarks are normalized and passed to the model to predict the corresponding sign language character.
The prediction is displayed on the video frame with a bounding box around the detected hand.

5. sign_language.py:

This is a GUI application built with Tkinter for sign language detection.
It prompts the user to start a sign detection session.
Upon starting, it opens the webcam and uses MediaPipe and the trained model to detect and predict sign language characters in real-time.
The predicted character is displayed on the video feed and printed in the console.
The user can quit the session by pressing 'q'.

# Overall Flow:

* The user first collects images for each sign class using create_img.py.
* Then, create_dataset.py processes these images to extract hand landmarks and prepares the dataset.
* train_classifier.py trains a machine learning model on this dataset.
* inference_classifier.py uses the trained model to predict signs in real-time from webcam input.
* sign_language.py provides a user-friendly GUI to run the real-time sign language detection using the trained model.
* This modular structure separates data collection, preprocessing, training, inference, and user interaction into distinct scripts, making the project organized and easy to follow.

# To run the project follow the steps:
1. Open Terminal
2. Install required modules:
* mediapipe (pip install mediapipe)
* opencv-python (pip install opencv-python)
* matplotlib (pip install matplotlib)
* numpy (pip install numpy)
* scikit-learn (pip install scikit-learn)
* tkinter (usually comes pre-installed with Python, but on some systems you may need to install it separately)
3. Run The file sequense wise: create_img.py, create_dataset.py, inference_classifier.py, sign_language.py

# Sign Language Image
![WhatsApp Image 2024-09-13 at 15 14 41_31ecc5da](https://github.com/user-attachments/assets/68540041-289e-4148-9277-5ed8b9f549b2)
