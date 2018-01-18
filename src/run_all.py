### 1. Import libraries and modules. Load env variables

# 1.1 Standard library importsimport sys
import os
import sys

# 1.2 Related third party imports
import numpy as np
from matplotlib import pyplot
from dotenv import find_dotenv, load_dotenv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# 1.3 Local application/library specific imports
from data.make_dataset import *

# Initial setup
np.random.seed(123)  # For reproducibility
load_dotenv(find_dotenv()) # Reads the key,value pair from .env and adds them to env. vars 

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

if __name__ == '__main__':

### 2. Download raw data sets
    filename = "PRSA_data_2010.1.1-2014.12.31.csv" #Caution! Check url for last filename
    dataset_name = download_new_dataset(filename)
    copy_new_downloaded_dataset(dataset_name, filename) #Put last dataset in position to source the following flowchart
    
### 3. Preprocess input data
    X_train, y_train, X_test, y_test = make_dataset(filename)

### 4. Preprocess class labels for Keras
    #Y_train = np_utils.to_categorical(y_train, 10)
    #Y_test = np_utils.to_categorical(y_test, 10)

 
### 5. Define model architecture
model = Sequential()
 
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1)) 
 
### 6. Compile model
model.compile(loss='mae',
              optimizer='adam')
 
### 7. Fit model on training data
history = model.fit(X_train, y_train,
                    epochs=50, batch_size=72,
                    validation_data=(X_test, y_test),
                    verbose=2, shuffle=False)

### 8. Evaluate model on test data
# score = model.evaluate(X_test, y_test, verbose=0)

### 9. Analysis and plotting
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

