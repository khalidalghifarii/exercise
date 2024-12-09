# %%
# Data visualization
import numpy as np
import pandas as pd 
# Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

# Train-Test
from sklearn.model_selection import train_test_split
# Classification Report
from sklearn.metrics import classification_report, confusion_matrix

import pickle

import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Describe dataset and Train Model

# %% [markdown]
# ### 1.1. Describe dataset

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


# Remove duplicate rows (optional)
def remove_duplicate_rows(dataset_path: str):
    '''
    Remove duplicated data from the dataset then save it to another files
    '''
    
    df = pd.read_csv(dataset_path)
    df.drop_duplicates(keep="first", inplace=True)
    df.to_csv(f"cleaned_dataset.csv", sep=',', encoding='utf-8', index=False)


df = describe_dataset("./dataset.csv")

# %% [markdown]
# ### 1.2. Preprocess data

# %%
# load dataset
df = pd.read_csv("./dataset.csv")

# Categorizing label
df.loc[df["label"] == "I", "label"] = 0
df.loc[df["label"] == "M", "label"] = 1
df.loc[df["label"] == "D", "label"] = 2

print(f'Number of rows: {df.shape[0]} \nNumber of columns: {df.shape[1]}\n')
print(f"Labels: \n{df['label'].value_counts()}\n")

# %%
# Standard Scaling of features
with open("./model/input_scaler.pkl", "rb") as f2:
    input_scaler = pickle.load(f2)

x = df.drop("label", axis = 1)
x = pd.DataFrame(input_scaler.transform(x))

y = df["label"]

# # Converting prediction to categorical
y_cat = to_categorical(y)

# %%
x_train, x_test, y_train, y_test = train_test_split(x.values, y_cat, test_size=0.2)

# %% [markdown]
# ### 1.3. Train model

# %%
model = Sequential()
model.add(Dense(52, input_dim = 52, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(52, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(14, activation = "relu"))
model.add(Dense(3, activation = "softmax"))
model.compile(Adam(lr = 0.01), "categorical_crossentropy", metrics = ["accuracy"])
model.summary()

# %%
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_data=(x_test, y_test))

# %% [markdown]
# ## 2. Model Evaluation

# %% [markdown]
# ### 2.1. Train set evaluation

# %%
predict_x = model.predict(x_test) 
y_pred_class = np.argmax(predict_x, axis=1)

y_pred = model.predict(x_test)
y_test_class = np.argmax(y_test, axis=1)

confusion_matrix(y_test_class, y_pred_class)

# %%
print(classification_report(y_test_class, y_pred_class))

# %% [markdown]
# ### 2.2. Test set evaluation

# %%
test_df = pd.read_csv("./test.csv")

# Categorizing label
test_df.loc[test_df["label"] == "I", "label"] = 0
test_df.loc[test_df["label"] == "M", "label"] = 1
test_df.loc[test_df["label"] == "D", "label"] = 2

# %%
# Standard Scaling of features
test_x = test_df.drop("label", axis = 1)
test_x = pd.DataFrame(input_scaler.transform(test_x))

test_y = test_df["label"]

# # Converting prediction to categorical
test_y_cat = to_categorical(test_y)

# %%
predict_x = model.predict(test_x) 
y_pred_class = np.argmax(predict_x, axis=1)
y_test_class = np.argmax(test_y_cat, axis=1)

confusion_matrix(y_test_class, y_pred_class)

# %%
print(classification_report(y_test_class, y_pred_class))

# %% [markdown]
# ## 3. Dump Model

# %%
# Dump the best model to a pickle file
with open("./model/lunge_model_deep_learning.pkl", "wb") as f:
    pickle.dump(model, f)

# %%



