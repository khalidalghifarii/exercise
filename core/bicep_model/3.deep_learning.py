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
from keras.callbacks import EarlyStopping, TensorBoard
import keras_tuner as kt

# Train-Test
from sklearn.model_selection import train_test_split
# Classification Report
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import pickle

import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Important Landmarks and Important functions

# %%
# Determine important landmarks for plank
IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "LEFT_ELBOW",
    "RIGHT_WRIST",
    "LEFT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
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
    df.to_csv(f"cleaned_train.csv", sep=',', encoding='utf-8', index=False)


def round_up_metric_results(results) -> list:
    '''Round up metrics results such as precision score, recall score, ...'''
    return list(map(lambda el: round(el, 3), results))

# %% [markdown]
# ## 2. Describe Dataset & Split Data

# %%
# load dataset
df = describe_dataset("./train.csv")

# Categorizing label
df.loc[df["label"] == "C", "label"] = 0
df.loc[df["label"] == "L", "label"] = 1

# %%
with open("./model/input_scaler.pkl", "rb") as f:
    sc = pickle.load(f)

# %%
# Standard Scaling of features
x = df.drop("label", axis = 1)
x = pd.DataFrame(sc.transform(x))

y = df["label"]

# # Converting prediction to categorical
y_cat = to_categorical(y)

# %%
x_train, x_test, y_train, y_test = train_test_split(x.values, y_cat, test_size=0.2, random_state=1234)

# %% [markdown]
# ## 3. Build Model

# %% [markdown]
# ### 3.1. Set up

# %%
stop_early = EarlyStopping(monitor='val_loss', patience=3)

# Final Results
final_models = {}

# %%
def describe_model(model):
    '''
    Describe Model architecture
    '''
    print(f"Describe models architecture")
    for i, layer in enumerate(model.layers):
        number_of_units = layer.units if hasattr(layer, 'units') else 0

        if hasattr(layer, "activation"):
            print(f"Layer-{i + 1}: {number_of_units} units, func: ", layer.activation)
        else:
            print(f"Layer-{i + 1}: {number_of_units} units, func: None")


def get_best_model(tuner):
    '''
    Describe and return the best model found from keras tuner
    '''
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)

    describe_model(best_model)

    print("\nOther params:")
    ignore_params = ["tuner", "activation", "layer", "epoch"]
    for param, value in best_hps.values.items():
        if not any(word in param for word in ignore_params):
            print(f"{param}: {value}")

    return best_model

# %% [markdown]
# ### 3.2. Model with 3 layers

# %%
def model_3l_builder(hp):
    model = Sequential()
    model.add(Dense(36, input_dim = 36, activation = "relu"))

    hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
    hp_layer_1 = hp.Int('layer_1', min_value=32, max_value=512, step=32)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.add(Dense(units=hp_layer_1, activation=hp_activation))
    model.add(Dense(2, activation = "softmax"))

    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss="categorical_crossentropy", metrics = ["accuracy"])
    
    return model

# %%
tuner_3l = kt.Hyperband(
    model_3l_builder,
    objective='val_accuracy',
    max_epochs=10,
    directory='keras_tuner_dir',
    project_name='keras_tuner_demo',
)
tuner_3l.search(x_train, y_train, validation_data=(x_test, y_test), epochs=10, callbacks=[stop_early])

# %%
model_3l = get_best_model(tuner_3l)
model_3l.fit(x_train, y_train, epochs=100, batch_size=10, validation_data=(x_test, y_test), callbacks=[stop_early])

# %%
final_models["3_layers"] = model_3l

# %% [markdown]
# ### 3.3. Model with 5 layers

# %%
def model_5l_builder(hp):
    model = Sequential()
    model.add(Dense(36, input_dim = 36, activation = "relu"))

    hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
    hp_layer_1 = hp.Int('layer_1', min_value=32, max_value=512, step=32)
    hp_layer_2 = hp.Int('layer_2', min_value=32, max_value=512, step=32)
    hp_layer_3 = hp.Int('layer_3', min_value=32, max_value=512, step=32)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.add(Dense(units=hp_layer_1, activation=hp_activation))
    model.add(Dense(units=hp_layer_2, activation=hp_activation))
    model.add(Dense(units=hp_layer_3, activation=hp_activation))
    model.add(Dense(2, activation = "softmax"))

    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss="categorical_crossentropy", metrics = ["accuracy"])
    
    return model

# %%
tuner_5l = kt.Hyperband(
    model_5l_builder,
    objective='val_accuracy',
    max_epochs=10,
    directory='keras_tuner_dir',
    project_name='keras_tuner_demo_2'
)
tuner_5l.search(x_train, y_train, validation_data=(x_test, y_test), epochs=10, callbacks=[stop_early, TensorBoard("./keras_tuner_dir/logs")])

# %%
model_5l = get_best_model(tuner_5l)
model_5l.fit(x_train, y_train, epochs=100, batch_size=10, validation_data=(x_test, y_test), callbacks=[stop_early])

# %%
final_models["5_layers"] = model_5l

# %% [markdown]
# ### 3.4. Model with 7 layers include Dropout

# %%
def model_7lD_builder(hp):
    model = Sequential()
    model.add(Dense(36, input_dim = 36, activation = "relu"))

    hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
    hp_layer_1 = hp.Int('layer_1', min_value=32, max_value=512, step=32)
    hp_layer_2 = hp.Int('layer_2', min_value=32, max_value=512, step=32)
    hp_layer_3 = hp.Int('layer_3', min_value=32, max_value=512, step=32)
    hp_dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
    hp_dropout_2 = hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.add(Dense(units=hp_layer_1, activation=hp_activation))
    model.add(Dropout(rate=hp_dropout_1))
    model.add(Dense(units=hp_layer_2, activation=hp_activation))
    model.add(Dropout(rate=hp_dropout_2))
    model.add(Dense(units=hp_layer_3, activation=hp_activation))
    model.add(Dense(2, activation = "softmax"))

    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss="categorical_crossentropy", metrics = ["accuracy"])
    
    return model

# %%
tuner_7lD = kt.Hyperband(
    model_7lD_builder,
    objective='accuracy',
    max_epochs=10,
    directory='keras_tuner_dir',
    project_name='keras_tuner_demo_3'
)
tuner_7lD.search(x_train, y_train, validation_data=(x_test, y_test), epochs=10, callbacks=[stop_early, TensorBoard("./keras_tuner_dir/logs")])

# %%
model_7lD = get_best_model(tuner_7lD)
model_7lD.fit(x_train, y_train, epochs=100, batch_size=10, validation_data=(x_test, y_test), callbacks=[stop_early])

# %%
final_models["7_layers_with_dropout"] = model_7lD

# %% [markdown]
# ### 3.5. Model with 7 layers

# %%
def model_7l_builder(hp):
    model = Sequential()
    model.add(Dense(36, input_dim = 36, activation = "relu"))

    hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
    hp_layer_1 = hp.Int('layer_1', min_value=32, max_value=512, step=32)
    hp_layer_2 = hp.Int('layer_2', min_value=32, max_value=512, step=32)
    hp_layer_3 = hp.Int('layer_3', min_value=32, max_value=512, step=32)
    hp_layer_4 = hp.Int('layer_4', min_value=32, max_value=512, step=32)
    hp_layer_5 = hp.Int('layer_5', min_value=32, max_value=512, step=32)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.add(Dense(units=hp_layer_1, activation=hp_activation))
    model.add(Dense(units=hp_layer_2, activation=hp_activation))
    model.add(Dense(units=hp_layer_3, activation=hp_activation))
    model.add(Dense(units=hp_layer_4, activation=hp_activation))
    model.add(Dense(units=hp_layer_5, activation=hp_activation))
    model.add(Dense(2, activation = "softmax"))

    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss="categorical_crossentropy", metrics = ["accuracy"])
    
    return model

# %%
tuner_7l = kt.Hyperband(
    model_7l_builder,
    objective='val_accuracy',
    max_epochs=10,
    directory='keras_tuner_dir',
    project_name='keras_tuner_demo_6'
)
tuner_7l.search(x_train, y_train, validation_data=(x_test, y_test), epochs=10, callbacks=[stop_early, TensorBoard("./keras_tuner_dir/logs")])

# %%
model_7l = get_best_model(tuner_7l)
model_7l.fit(x_train, y_train, epochs=100, batch_size=10, validation_data=(x_test, y_test), callbacks=[stop_early])

# %%
final_models["7_layers"] = model_7l

# %% [markdown]
# ### 3.6. Describe final models

# %%
for name, model in final_models.items():
    print(f"{name}: ", end="")
    describe_model(model)
    print()

# %% [markdown]
# ## 4. Model Evaluation

# %% [markdown]
# ### 4.1. Train set evaluation

# %%
train_set_results = []

for name, model in final_models.items():
    # Evaluate model
    predict_x = model.predict(x_test, verbose=False) 
    y_pred_class = np.argmax(predict_x, axis=1)
    y_test_class = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_test_class, y_pred_class, labels=[0, 1])
    (p_score, r_score, f_score, _) = precision_recall_fscore_support(y_test_class, y_pred_class, labels=[0, 1])
    
    train_set_results.append(( name, round_up_metric_results(p_score), round_up_metric_results(r_score), round_up_metric_results(f_score), cm ))

train_set_results.sort(key=lambda k: sum(k[3]), reverse=True)
pd.DataFrame(train_set_results, columns=["Model", "Precision Score", "Recall Score", "F1 score", "Confusion Matrix"])

# %% [markdown]
# ### 4.2. Test set evaluation

# %%
# load dataset
test_df = describe_dataset("./test.csv")

# Categorizing label
test_df.loc[test_df["label"] == "C", "label"] = 0
test_df.loc[test_df["label"] == "L", "label"] = 1

# %%
# Standard Scaling of features
test_x = test_df.drop("label", axis = 1)
test_x = pd.DataFrame(sc.transform(test_x))

test_y = test_df["label"]

# # Converting prediction to categorical
test_y_cat = to_categorical(test_y)

# %%
test_set_results = []

for name, model in final_models.items():
    # Evaluate model
    predict_x = model.predict(test_x, verbose=False) 
    y_pred_class = np.argmax(predict_x, axis=1)
    y_test_class = np.argmax(test_y_cat, axis=1)

    cm = confusion_matrix(y_test_class, y_pred_class, labels=[0, 1])
    (p_score, r_score, f_score, _) = precision_recall_fscore_support(y_test_class, y_pred_class, labels=[0, 1])
    
    test_set_results.append(( name, round_up_metric_results(p_score), round_up_metric_results(r_score), round_up_metric_results(f_score), cm ))

test_set_results.sort(key=lambda k: k[1] + k[2] + k[3], reverse=True)
pd.DataFrame(test_set_results, columns=["Model", "Precision Score", "Recall Score", "F1 score", "Confusion Matrix"])

# %% [markdown]
# ## 5. Dumped Model

# %%
# Dump the best model to a pickle file
with open("./model/bicep_dp.pkl", "wb") as f:
    pickle.dump(final_models["7_layers"], f)

# %%
# Dump final results
with open("./model/all_models.pkl", "wb") as f:
    pickle.dump(final_models, f)

# %%



