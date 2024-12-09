# %%
import mediapipe as mp
import cv2
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# %% [markdown]
# ### 1. Train Model

# %% [markdown]
# #### 1.1. Describe data and split dataset

# %%
def rescale_frame(frame, percent=50):
    '''
    Rescale a frame to a certain percentage compare to its original frame
    '''
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)


def describe_dataset(dataset_path: str):
    '''
    Describe dataset
    '''

    data = pd.read_csv(dataset_path)
    print(f"Headers: {list(data.columns.values)}")
    print(f'Number of rows: {data.shape[0]} \nNumber of columns: {data.shape[1]}\n')
    print(f"Labels: \n{data['label'].value_counts()}\n")
    print(f"Missing values: {data.isnull().values.any()}\n")
    
    duplicate = data[data.duplicated()]
    print(f"Duplicate Rows : {len(duplicate.sum(axis=1))}")

    return data


def round_up_metric_results(results) -> list:
    '''Round up metrics results such as precision score, recall score, ...'''
    return list(map(lambda el: round(el, 3), results))

# %%
df = describe_dataset("./train.csv")
df.loc[df["label"] == "C", "label"] = 0
df.loc[df["label"] == "H", "label"] = 1
df.loc[df["label"] == "L", "label"] = 2
df.tail(3)

# %%
# Extract features and class
X = df.drop("label", axis=1)
y = df["label"].astype("int")

# %%
sc = StandardScaler()
X = pd.DataFrame(sc.fit_transform(X))

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
y_test.head(3)

# %% [markdown]
# #### 1.2. Train model using Scikit-Learn and train set evaluation

# %%
algorithms =[("LR", LogisticRegression()),
         ("SVC", SVC(probability=True)),
         ('KNN',KNeighborsClassifier()),
         ("DTC", DecisionTreeClassifier()),
         ("SGDC", CalibratedClassifierCV(SGDClassifier())),
         ("NB", GaussianNB()),
         ('RF', RandomForestClassifier()),]

models = {}
final_results = []

for name, model in algorithms:
    trained_model = model.fit(X_train, y_train)
    models[name] = trained_model

    # Evaluate model
    model_results = model.predict(X_test)

    p_score = precision_score(y_test, model_results, average=None, labels=[0, 1, 2])
    a_score = accuracy_score(y_test, model_results)
    r_score = recall_score(y_test, model_results, average=None, labels=[0, 1, 2])
    f1_score_result = f1_score(y_test, model_results, average=None, labels=[0, 1, 2])
    cm = confusion_matrix(y_test, model_results, labels=[0, 1, 2])
    final_results.append(( name,  round_up_metric_results(p_score), a_score, round_up_metric_results(r_score), round_up_metric_results(f1_score_result), cm))


# %%
# Sort results by F1 score
final_results.sort(key=lambda k: sum(k[4]), reverse=True)

pd.DataFrame(final_results, columns=["Model", "Precision Score", "Accuracy score", "Recall Score", "F1 score", "Confusion Matrix"])

# %% [markdown]
# #### 1.3. Test set evaluation

# %%
test_df = describe_dataset("./test.csv")
test_df = test_df.sample(frac=1).reset_index(drop=True)

test_df.loc[test_df["label"] == "C", "label"] = 0
test_df.loc[test_df["label"] == "H", "label"] = 1
test_df.loc[test_df["label"] == "L", "label"] = 2

test_x = test_df.drop("label", axis=1)
test_y = test_df["label"].astype("int")

test_x = pd.DataFrame(sc.transform(test_x))

# %%
testset_final_results = []

for name, model in models.items():
    # Evaluate model
    model_results = model.predict(test_x)

    p_score = precision_score(test_y, model_results, average=None, labels=[0, 1, 2])
    a_score = accuracy_score(test_y, model_results)
    r_score = recall_score(test_y, model_results, average=None, labels=[0, 1, 2])
    f1_score_result = f1_score(test_y, model_results, average=None, labels=[0, 1, 2])
    cm = confusion_matrix(test_y, model_results, labels=[0, 1, 2])
    testset_final_results.append(( name,  round_up_metric_results(p_score), a_score, round_up_metric_results(r_score), round_up_metric_results(f1_score_result), cm ))


testset_final_results.sort(key=lambda k: sum(k[4]), reverse=True)
pd.DataFrame(testset_final_results, columns=["Model", "Precision Score", "Accuracy score", "Recall Score", "F1 score", "Confusion Matrix"])

# %% [markdown]
# #### 1.4. Dumped model and input scaler using pickle
# 
# According to the evaluations, there are multiple good models at the moment, therefore, the best models are LR and Ridge.

# %%
with open("./model/all_sklearn.pkl", "wb") as f:
    pickle.dump(models, f)

# %%
with open("./model/LR_model.pkl", "wb") as f:
    pickle.dump(models["LR"], f)

# %%
with open("./model/SVC_model.pkl", "wb") as f:
    pickle.dump(models["SVC"], f)

# %%
# Dump input scaler
with open("./model/input_scaler.pkl", "wb") as f:
    pickle.dump(sc, f)


