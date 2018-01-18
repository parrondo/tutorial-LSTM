# 1.1 Standard library importsimport sys
import os
import sys
import errno
import h5py
import cv2
import glob
from datetime import datetime

# 1.2 Related third party imports
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import numpy as np
import parmap
from shutil import copyfile
from dotenv import find_dotenv, load_dotenv
sys.path.append(os.path.join(os.path.dirname(__file__),"../app"))
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# 1.3 Local application/library specific imports
# None


def copy_new_downloaded_dataset(dataset_name, filename):

    """
    #Copy downloaded data file from source to target destination
    args:
         source (str) source file with full path
         target (str) target file with full path
    yields:
         new file from source to target
    """
    raw_dir = os.environ.get("RAW_DIR")
    source = os.path.join(raw_dir, dataset_name, filename)
    target = os.path.join(raw_dir, filename)

    # adding exception handling
    try:
        copyfile(source, target)
    except IOError as e:
        print("Unable to copy file. %s" % e)
        exit(1)
    except:
        print("Unexpected error:", sys.exc_info())
        exit(1)
 
    print("\nFile copy done!\n")


def download_new_dataset(filename):

    """
    Download last dataset from source
    args: None
    yields:
           dataset_name (str) Name of the downloaded dataset directory
    """

    # set up download path with actual date
    RAW_DIR = os.path.expanduser(os.environ.get("RAW_DIR"))
    from datetime import datetime
    now = datetime.now().strftime("%Y%m%d")
    dataset_name = "dataset_{0}".format(now) #Name of dataset directory
    print("Downloading the current dataset...")
    url = os.path.join("https://archive.ics.uci.edu/ml/machine-learning-databases/00381/", filename)
    target = os.path.join(RAW_DIR, dataset_name, filename)
    r = requests.get(url, allow_redirects=True)
    
    #Granting file folder is created
    if not os.path.exists(os.path.dirname(target)):
        try:
            os.makedirs(os.path.dirname(target))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(target, "wb") as f:
        f.write(r.content)

    return dataset_name


def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    """
    Convert series to supervised learning. Frame a time series as a supervised learning dataset.
    args:
        data (list or NumPy array) Sequence of observations.
        n_in (int) Number of lag observations as input (X).
        n_ou: (int)  Boolean whether or not to drop rows with NaN values.
    yields:
        agg (Pandas DataFrame) of series framed for supervised learning.
    """

    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg


def clean_data(filename):

    """
    Clean data and parse dates
    args:
         filename (str) filename for dataset to clean
    """

    #Directories
    raw_dir = os.path.expanduser(os.environ.get("RAW_DIR"))
    interim_dir = os.path.expanduser(os.environ.get("INTERIM_DIR"))
    #Full path filenames
    raw_data = os.path.join(raw_dir, filename)
    interim_data = os.path.join(interim_dir,"pollution.csv")
    # load data
    dataset = read_csv(raw_data,  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    dataset.drop('No', axis=1, inplace=True)
    # manually specify column names
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'
    # mark all NA values with 0
    dataset['pollution'].fillna(0, inplace=True)
    # drop the first 24 hours
    dataset = dataset[24:]
    # summarize first 5 rows
    print(dataset.head(5))
    # save to interim data file
    dataset.to_csv(interim_data)
    print("Raw data cleaned yet")

    return


def make_dataset(filename):

    """
    Generate interim and processed data files from raw data file
    args: 
          filename (str) raw data filename. This file must be into the raw folder.
          training_data (str) the file with training data
          tournament_data (str) the file with both validation and test data
          processed_dir (str) directory for processed data files
    yields :
           the following files into PROCESSED_DIR:
           x_train.csv, y_train.csv, x_val.csv, y_val.csv, x_test.csv         
    """

    #Directories
    raw_dir = os.path.expanduser(os.environ.get("RAW_DIR"))
    interim_dir = os.path.expanduser(os.environ.get("INTERIM_DIR"))
    processed_dir = os.path.expanduser(os.environ.get("PROCESSED_DIR"))
    #Full path filenames
    raw_data = os.path.join(raw_dir, filename)
    interim_data = os.path.join(interim_dir,"pollution.csv")
    #Clean data
    clean_data(filename)
    # load dataset
    dataset = read_csv(interim_data, header=0, index_col=0)
    values = dataset.values
    # integer encode direction
    encoder = LabelEncoder()
    values[:,4] = encoder.fit_transform(values[:,4])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
    print(reframed.head())

    # split into train and test sets
    values = reframed.values
    n_years = 3
    n_train_hours = 365 * 24 * n_years
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print("Raw data processed yet")

    return X_train, y_train, X_test, y_test
 

if __name__ == '__main__':

    make_dataset()
