### 1. Import libraries and modules. Load env variables
import os
import sys
import numpy as np
np.random.seed(123)  # for reproducibility
from dotenv import find_dotenv, load_dotenv
#Reads the key,value pair from .env and adds them to environment variable 
load_dotenv(find_dotenv())
from data.make_dataset import *

# Check the env variables exist
raw_msg = "Set your raw data absolute path in the .env file at project root"
data_msg = "Set your processed data absolute path in the .env file at project root"
proj_msg = "Set your project absolute path in the .env file at project root"

assert "RAW_DIR" in os.environ, raw_msg
assert "DATA_DIR" in os.environ, data_msg
assert "PROJ_DIR" in os.environ, proj_msg

# Load env variables (path to each kind of file)
PROJ_DIR = os.path.expanduser(os.environ.get("PROJ_DIR"))
RAW_DIR = os.path.expanduser(os.environ.get("RAW_DIR"))
DATA_DIR = os.path.expanduser(os.environ.get("DATA_DIR"))
PROCESSED_DIR = os.path.expanduser(os.environ.get("PROCESSED_DIR"))
FIGURES_DIR = os.path.expanduser(os.environ.get("FIGURES_DIR"))
MODEL_DIR = os.path.expanduser(os.environ.get("MODEL_DIR"))
EXTERNAL_DIR = os.path.expanduser(os.environ.get("EXTERNAL_DIR"))
#Original numer.ai data files (inmutables raw files)
TRAINING_DATA = os.path.join(RAW_DIR, "numerai_training_data.csv")
TOURNAMENT_DATA = os.path.join(RAW_DIR, "numerai_tournament_data.csv")


if __name__ == '__main__':

### 2. Download raw data sets
    filename = "PRSA_data_2010.1.1-2014.12.31.csv" #Caution! Check url for last filename
    dataset_name = download_new_dataset(filename)
    copy_new_downloaded_dataset(dataset_name, filename) #Put last dataset in position to source the following flowchart
    
### 3. Preprocess input data
    make_dataset(filename)

### 4. Preprocess class labels for Keras
    #Y_train = np_utils.to_categorical(y_train, 10)
    #Y_test = np_utils.to_categorical(y_test, 10)

""" 
### 5. Define model architecture
model = Sequential()
 
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
 
### 6. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
### 7. Fit model on training data
model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=10, verbose=1)
 
### 8. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
"""
