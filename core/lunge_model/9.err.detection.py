# %%
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import traceback
import pickle

import warnings
warnings.filterwarnings('ignore')

# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# %% [markdown]
# ## 1. Setup important landmarks and functions

# %%
# Determine important landmarks for lunge
IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
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


def calculate_angle(point1: list, point2: list, point3: list) -> float:
    '''
    Calculate the angle between 3 points
    Unit of the angle will be in Degree
    '''
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)

    # Calculate algo
    angleInRad = np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(point1[1] - point2[1], point1[0] - point2[0])
    angleInDeg = np.abs(angleInRad * 180.0 / np.pi)

    angleInDeg = angleInDeg if angleInDeg <= 180 else 360 - angleInDeg
    return angleInDeg
    

def analyze_knee_angle(
    mp_results, stage: str, angle_thresholds: list, draw_to_image: tuple = None
):
    """
    Calculate angle of each knee while performer at the DOWN position

    Return result explanation:
        error: True if at least 1 error
        right
            error: True if an error is on the right knee
            angle: Right knee angle
        left
            error: True if an error is on the left knee
            angle: Left knee angle
    """
    results = {
        "error": None,
        "right": {"error": None, "angle": None},
        "left": {"error": None, "angle": None},
    }

    landmarks = mp_results.pose_landmarks.landmark

    # Calculate right knee angle
    right_hip = [
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
    ]
    right_knee = [
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
    ]
    right_ankle = [
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
    ]
    results["right"]["angle"] = calculate_angle(right_hip, right_knee, right_ankle)

    # Calculate left knee angle
    left_hip = [
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
    ]
    left_knee = [
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
    ]
    left_ankle = [
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
    ]
    results["left"]["angle"] = calculate_angle(left_hip, left_knee, left_ankle)

    # Draw to image
    if draw_to_image is not None and stage != "down":
        (image, video_dimensions) = draw_to_image

        # Visualize angles
        cv2.putText(
            image,
            str(int(results["right"]["angle"])),
            tuple(np.multiply(right_knee, video_dimensions).astype(int)),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(int(results["left"]["angle"])),
            tuple(np.multiply(left_knee, video_dimensions).astype(int)),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if stage != "down":
        return results

    # Evaluation
    results["error"] = False

    if angle_thresholds[0] <= results["right"]["angle"] <= angle_thresholds[1]:
        results["right"]["error"] = False
    else:
        results["right"]["error"] = True
        results["error"] = True

    if angle_thresholds[0] <= results["left"]["angle"] <= angle_thresholds[1]:
        results["left"]["error"] = False
    else:
        results["left"]["error"] = True
        results["error"] = True

    # Draw to image
    if draw_to_image is not None:
        (image, video_dimensions) = draw_to_image

        right_color = (255, 255, 255) if not results["right"]["error"] else (0, 0, 255)
        left_color = (255, 255, 255) if not results["left"]["error"] else (0, 0, 255)

        right_font_scale = 0.5 if not results["right"]["error"] else 1
        left_font_scale = 0.5 if not results["left"]["error"] else 1

        right_thickness = 1 if not results["right"]["error"] else 2
        left_thickness = 1 if not results["left"]["error"] else 2

        # Visualize angles
        cv2.putText(
            image,
            str(int(results["right"]["angle"])),
            tuple(np.multiply(right_knee, video_dimensions).astype(int)),
            cv2.FONT_HERSHEY_COMPLEX,
            right_font_scale,
            right_color,
            right_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(int(results["left"]["angle"])),
            tuple(np.multiply(left_knee, video_dimensions).astype(int)),
            cv2.FONT_HERSHEY_COMPLEX,
            left_font_scale,
            left_color,
            left_thickness,
            cv2.LINE_AA,
        )

    return results

# %% [markdown]
# ## 2. Constants

# %%
# VIDEO_PATH1 = "../data/lunge/lunge_test_3.mp4"
# VIDEO_PATH2 = "../data/lunge/lunge_test_5.mp4"
VIDEO_PATH = "../../demo/lunge_demo.mp4"


# %%
with open("./model/input_scaler.pkl", "rb") as f:
    input_scaler = pickle.load(f)

# %% [markdown]
# ## 3. Detection with Sklearn Models

# %%
# Load model
with open("./model/sklearn/stage_SVC_model.pkl", "rb") as f:
    stage_sklearn_model = pickle.load(f)

with open("./model/sklearn/err_LR_model.pkl", "rb") as f:
    err_sklearn_model = pickle.load(f)

# %%
cap = cv2.VideoCapture(VIDEO_PATH)
current_stage = ""
counter = 0

prediction_probability_threshold = 0.8
ANGLE_THRESHOLDS = [60, 135]

knee_over_toe = False

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break

        # Reduce size of a frame
        image = rescale_frame(image, 50)
        video_dimensions = [image.shape[1], image.shape[0]]

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
            stage_predicted_class = stage_sklearn_model.predict(X)[0]
            stage_prediction_probabilities = stage_sklearn_model.predict_proba(X)[0]
            stage_prediction_probability = round(stage_prediction_probabilities[stage_prediction_probabilities.argmax()], 2)

            # Evaluate model prediction
            if stage_predicted_class == "I" and stage_prediction_probability >= prediction_probability_threshold:
                current_stage = "init"
            elif stage_predicted_class == "M" and stage_prediction_probability >= prediction_probability_threshold: 
                current_stage = "mid"
            elif stage_predicted_class == "D" and stage_prediction_probability >= prediction_probability_threshold:
                if current_stage in ["mid", "init"]:
                    counter += 1
                
                current_stage = "down"
            
            # Error detection
            # Knee angle
            analyze_knee_angle(mp_results=results, stage=current_stage, angle_thresholds=ANGLE_THRESHOLDS, draw_to_image=(image, video_dimensions))

            # Knee over toe
            err_predicted_class = err_prediction_probabilities = err_prediction_probability = None
            if current_stage == "down":
                err_predicted_class = err_sklearn_model.predict(X)[0]
                err_prediction_probabilities = err_sklearn_model.predict_proba(X)[0]
                err_prediction_probability = round(err_prediction_probabilities[err_prediction_probabilities.argmax()], 2)
                
            
            # Visualization
            # Status box
            cv2.rectangle(image, (0, 0), (800, 45), (245, 117, 16), -1)

            # Display stage prediction
            cv2.putText(image, "STAGE", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(stage_prediction_probability), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, current_stage, (50, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Display error prediction
            cv2.putText(image, "K_O_T", (200, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(err_prediction_probability), (195, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, str(err_predicted_class), (245, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Display Counter
            cv2.putText(image, "COUNTER", (110, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (110, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            break
        
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
# ## 4. Detection with deep learning model

# %%
# Load model
with open("./model/dp/err_lunge_dp.pkl", "rb") as f:
    err_deep_learning_model = pickle.load(f)

# %%
cap = cv2.VideoCapture(VIDEO_PATH)
current_stage = ""
counter = 0

prediction_probability_threshold = 0.8
ANGLE_THRESHOLDS = [60, 135]

knee_over_toe = False

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break

        # Reduce size of a frame
        image = rescale_frame(image, 50)
        video_dimensions = [image.shape[1], image.shape[0]]

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
            stage_predicted_class = stage_sklearn_model.predict(X)[0]
            stage_prediction_probabilities = stage_sklearn_model.predict_proba(X)[0]
            stage_prediction_probability = round(stage_prediction_probabilities[stage_prediction_probabilities.argmax()], 2)

            # Evaluate model prediction
            if stage_predicted_class == "I" and stage_prediction_probability >= prediction_probability_threshold:
                current_stage = "init"
            elif stage_predicted_class == "M" and stage_prediction_probability >= prediction_probability_threshold: 
                current_stage = "mid"
            elif stage_predicted_class == "D" and stage_prediction_probability >= prediction_probability_threshold:
                if current_stage == "mid":
                    counter += 1
                
                current_stage = "down"
            
            # Error detection
            # Knee angle
            analyze_knee_angle(mp_results=results, stage=current_stage, angle_thresholds=ANGLE_THRESHOLDS, draw_to_image=(image, video_dimensions))

            # Knee over toe
            err_predicted_class = err_prediction_probabilities = err_prediction_probability = None
            if current_stage == "down":
                err_prediction = err_deep_learning_model.predict(X, verbose=False)
                err_predicted_class = np.argmax(err_prediction, axis=1)[0]
                err_prediction_probability = round(max(err_prediction.tolist()[0]), 2)

                err_predicted_class = "C" if err_predicted_class == 1 else "L"
                
            
            # Visualization
            # Status box
            cv2.rectangle(image, (0, 0), (800, 45), (245, 117, 16), -1)

            # Display stage prediction
            cv2.putText(image, "STAGE", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(stage_prediction_probability), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, current_stage, (50, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Display error prediction
            cv2.putText(image, "K_O_T", (200, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(err_prediction_probability), (195, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, str(err_predicted_class), (245, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Display Counter
            cv2.putText(image, "COUNTER", (110, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (110, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            break
        
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
# ## 5. Conclusion
# 
# - For stage detection:
#     - Best Sklearn model: KNN
# - For error detection:
#     - Best Sklearn model: LR
#     - Both models are correct most of the time


