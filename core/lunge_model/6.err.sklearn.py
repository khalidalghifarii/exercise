# %%
import mediapipe as mp
import cv2
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV

import warnings
warnings.filterwarnings('ignore')

# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# %% [markdown]
# ## 1. Set up important functions

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

# %% [markdown]
# ## 2. Describe and process data

# %%
TRAIN_SET_PATH  = "./err.train.csv"
TEST_SET_PATH  = "./err.test.csv"

# %%
df = describe_dataset(TRAIN_SET_PATH)
# Categorizing label
df.loc[df["label"] == "L", "label"] = 0
df.loc[df["label"] == "C", "label"] = 1

df.tail(3)

# %%
with open("./model/input_scaler.pkl", "rb") as f:
    sc = pickle.load(f)

# %%
# Extract features and class
X = df.drop("label", axis=1)
y = df["label"].astype("int")

X = pd.DataFrame(sc.transform(X))

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
y_test.head(3)

# %% [markdown]
# ## 3. Train & Evaluate Model

# %% [markdown]
# ### 3.1. Train and evaluate model with train set

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

    p_score = precision_score(y_test, model_results, average=None, labels=[1, 0])
    a_score = accuracy_score(y_test, model_results)
    r_score = recall_score(y_test, model_results, average=None, labels=[1, 0])
    f1_score_result = f1_score(y_test, model_results, average=None, labels=[1, 0])
    cm = confusion_matrix(y_test, model_results, labels=[1, 0])
    final_results.append(( name,  round_up_metric_results(p_score), a_score, round_up_metric_results(r_score), round_up_metric_results(f1_score_result), cm))

# Sort results by F1 score
final_results.sort(key=lambda k: sum(k[4]), reverse=True)
pd.DataFrame(final_results, columns=["Model", "Precision Score", "Accuracy score", "Recall Score", "F1 score", "Confusion Matrix"])

# %% [markdown]
# ### 3.2. Test set evaluation

# %%
test_df = describe_dataset(TEST_SET_PATH)
test_df = test_df.sample(frac=1).reset_index(drop=True)

# Categorizing label
test_df.loc[test_df["label"] == "L", "label"] = 0
test_df.loc[test_df["label"] == "C", "label"] = 1

test_x = test_df.drop("label", axis=1)
test_y = test_df["label"].astype("int")

test_x = pd.DataFrame(sc.transform(test_x))

# %%
testset_final_results = []

for name, model in models.items():
    # Evaluate model
    model_results = model.predict(test_x)

    p_score = precision_score(test_y, model_results, average=None, labels=[1, 0])
    a_score = accuracy_score(test_y, model_results)
    r_score = recall_score(test_y, model_results, average=None, labels=[1, 0])
    f1_score_result = f1_score(test_y, model_results, average=None, labels=[1, 0])
    cm = confusion_matrix(test_y, model_results, labels=[1, 0])
    testset_final_results.append(( name,  round_up_metric_results(p_score), a_score, round_up_metric_results(r_score), round_up_metric_results(f1_score_result), cm ))


testset_final_results.sort(key=lambda k: sum(k[4]), reverse=True)
pd.DataFrame(testset_final_results, columns=["Model", "Precision Score", "Accuracy score", "Recall Score", "F1 score", "Confusion Matrix"])

# %% [markdown]
# ## 4. Dump Models 
# 
# According to the evaluation above, LR and KNN SGDC would be chosen for more eval.

# %%
with open("./model/sklearn/err_all_sklearn.pkl", "wb") as f:
    pickle.dump(models, f)

# %%
with open("./model/sklearn/err_SGDC_model.pkl", "wb") as f:
    pickle.dump(models["SGDC"], f)

# %%
with open("./model/sklearn/err_LR_model.pkl", "wb") as f:
    pickle.dump(models["LR"], f)

# %%



