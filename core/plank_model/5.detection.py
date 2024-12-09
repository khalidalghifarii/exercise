# %% [markdown]
# ## Make Detection with the Trained Model

# %%
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd

import pickle

import warnings
warnings.filterwarnings('ignore')

# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# %% [markdown]
# ### Reconstruct the input structure

# %%
# Determine important landmarks for plank
IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]

# Generate all columns of the data frame

HEADERS = ["label"] # Label column

for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

# %% [markdown]
# ### Setup some important functions

# %%
def extract_important_keypoints(results) -> list:
    '''
    Extract important keypoints from mediapipe pose detection
    '''
    landmarks = results.pose_landmarks.landmark

    data = []
    for lm in IMPORTANT_LMS:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    
    return np.array(data).flatten().tolist()


def rescale_frame(frame, percent=50):
    '''
    Rescale a frame to a certain percentage compare to its original frame
    '''
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

# %%
# VIDEO_PATH1 = "../data/plank/plank_test.mov"
# VIDEO_PATH2 = "../data/plank/plank_test_1.mp4"
# VIDEO_PATH3 = "../data/plank/plank_test_2.mp4"
# VIDEO_PATH4 = "../data/plank/plank_test_3.mp4"
# VIDEO_PATH5 = "../data/plank/plank_test_4.mp4"
VIDEO_TEST = "../../demo/plank_demo.mp4"

# %% [markdown]
# ## 1. Make detection with Scikit learn model

# %%
# Load model
with open("./model/LR_model.pkl", "rb") as f:
    sklearn_model = pickle.load(f)

# Load input scaler
with open("./model/input_scaler.pkl", "rb") as f2:
    input_scaler = pickle.load(f2)

# Transform prediction into class
def get_class(prediction: float) -> str:
    return {
        0: "C",
        1: "H",
        2: "L",
    }.get(prediction)


# %%
cap = cv2.VideoCapture(VIDEO_TEST)
current_stage = ""
prediction_probability_threshold = 0.6

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break

        # Reduce size of a frame
        image = rescale_frame(image, 50)
        # image = cv2.flip(image, 1)

        # Recolor image from BGR to RGB for mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        if not results.pose_landmarks:
            print("No human found")
            continue

        # Recolor image from BGR to RGB for mediapipe
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))

        # Make detection
        try:
            # Extract keypoints from frame for the input
            row = extract_important_keypoints(results)
            X = pd.DataFrame([row], columns=HEADERS[1:])
            X = pd.DataFrame(input_scaler.transform(X))

            # Make prediction and its probability
            predicted_class = sklearn_model.predict(X)[0]
            predicted_class = get_class(predicted_class)
            prediction_probability = sklearn_model.predict_proba(X)[0]
            # print(predicted_class, prediction_probability)

            # Evaluate model prediction
            if predicted_class == "C" and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold:
                current_stage = "Correct"
            elif predicted_class == "L" and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold: 
                current_stage = "Low back"
            elif predicted_class == "H" and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold: 
                current_stage = "High back"
            else:
                current_stage = "unk"
            
            # Visualization
            # Status box
            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

            # Display class
            cv2.putText(image, "CLASS", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, current_stage, (90, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display probability
            cv2.putText(image, "PROB", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(prediction_probability[np.argmax(prediction_probability)], 2)), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")
        
        cv2.imshow("CV2", image)
        
        # Press Q to close cv2 window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # (Optional)Fix bugs cannot close windows in MacOS (https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv)
    for i in range (1, 5):
        cv2.waitKey(1)
  

# %% [markdown]
# ## 2. Make detection with Deep Learning Model

# %%
# Load model
with open("./model/plank_dp.pkl", "rb") as f:
    deep_learning_model = pickle.load(f)

# %%
cap = cv2.VideoCapture(VIDEO_TEST)
current_stage = ""
prediction_probability_threshold = 0.6

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break

        # Reduce size of a frame
        image = rescale_frame(image, 50)

        # Recolor image from BGR to RGB for mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        if not results.pose_landmarks:
            print("No human found")
            continue

        # Recolor image from BGR to RGB for mediapipe
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))

        # Make detection
        try:
            # Extract keypoints from frame for the input
            row = extract_important_keypoints(results)
            X = pd.DataFrame([row, ], columns=HEADERS[1:])
            X = pd.DataFrame(input_scaler.transform(X))
            

            # Make prediction and its probability
            prediction = deep_learning_model.predict(X)
            predicted_class = np.argmax(prediction, axis=1)[0]

            prediction_probability = max(prediction.tolist()[0])
            # print(X)

            # Evaluate model prediction
            if predicted_class == 0 and prediction_probability >= prediction_probability_threshold:
                current_stage = "Correct"
            elif predicted_class == 2 and prediction_probability >= prediction_probability_threshold: 
                current_stage = "Low back"
            elif predicted_class == 1 and prediction_probability >= prediction_probability_threshold: 
                current_stage = "High back"
            else:
                current_stage = "Unknown"

            # Visualization
            # Status box
            cv2.rectangle(image, (0, 0), (550, 60), (245, 117, 16), -1)

            # # Display class
            cv2.putText(image, "DETECTION", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, current_stage, (90, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # # Display class
            cv2.putText(image, "CLASS", (350, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(predicted_class), (345, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # # Display probability
            cv2.putText(image, "PROB", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(prediction_probability, 2)), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")
        
        cv2.imshow("CV2", image)
        
        # Press Q to close cv2 window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # (Optional)Fix bugs cannot close windows in MacOS (https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv)
    for i in range (1, 5):
        cv2.waitKey(1)
  

# %%
X

# %%



