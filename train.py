import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 
from keras.models import model_from_json
import os
print(os.listdir("../input"))
import random
import sys
import io
import os
import glob
import IPython
import matplotlib.pyplot as plt
import gc
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D, Conv2D, Conv3D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam

import time
start_time = time.time()
kernel_timeout = start_time + 10000

sample_data=False
sample_per=100
gc.collect()

rowcount = 1000000

#this one is high def data    
if(not sample_data):
    #kernel supports >6m <60m 
    train = pd.read_csv("../input/train.csv"
                        ,dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32}
                        #, nrows=rowcount
                       ) 
    train.rename({"acoustic_data": "acd", "time_to_failure": "ttf"}, axis="columns", inplace=True)    
    rowcount=int(train.shape[0])

#this one reads all the input, but in np.float32 format.
if(sample_data):
    train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
    train.rename({"acoustic_data": "acd", "time_to_failure": "ttf"}, axis="columns", inplace=True)    
    acd_small = train['acd'].values[::sample_per]
    ttf_small = train['ttf'].values[::sample_per]
    del train
    gc.collect()
    acd=acd_small
    ttf=ttf_small
    
    rowcount=int(train.shape[0]/sample_per)
    
#May be changed later to row by row reading to avoid crashing on 6e8 datapoints.
#https://docs.python.org/2/library/csv.html

post_input_time = time.time()
print("Read the input in {0:.2f} seconds.".format(post_input_time-start_time))


train = train.values

#takes the segmented data frame and returns its features
def feature_generate(x):
    #data frame should have size 1500
    features = np.c_[x.mean(axis=1), x.min(axis=1), x.max(axis=1), x.std(axis=1)]
    return (features)

#prepares data to be given as input
def prepare_data(x, n_steps = 100, step_length = 1500):
    x = np.array(x)
    temp = x.reshape(n_steps,-1)
    return np.c_[feature_generate(temp), feature_generate(temp[:, -step_length//10:]),feature_generate(temp[:, -step_length//100:])]


n_features = prepare_data(train[0:150000]).shape[1]

#generator to be used for the model
def data_generator(data, start_index, end_index, n_features=n_features, batch_size=32, n_steps=100, step_length=1500):
    
    assert end_index - n_steps*step_length >= 0
    
    while True:
        rows = np.random.randint(start_index + n_steps * step_length, end_index, size=batch_size)
        
        samples = np.zeros((batch_size, n_steps, n_features))
        targets = np.zeros(batch_size,)
        
        for i, row in enumerate(rows):
            samples[i] = prepare_data(data[row-(n_steps * step_length):row,0])
          
            targets[i] = data[row-1,1]
        targets = np.reshape(targets,(batch_size,1,1))
        yield samples, targets


#ready the generators
split = train.shape[0]*9//10
train_gen = data_generator(train, start_index=0, end_index = split)
valid_gen = data_generator(train, start_index = split+1, end_index = train.shape[0]-1)

#model inspired by the trigger word detection model used by Andrew Ng in his coursera course.
#check below the model for sauce.

def model(input_shape):
    
    X_input = Input(shape = input_shape)
    
    # Step 1: CONV layer (≈4 lines)
    X = Conv1D(64, kernel_size=1, strides=1)(X_input)                                 # CONV1D
    X = BatchNormalization()(X)                                 # Batch normalization
    X = Activation('relu')(X)                                 # ReLu activation
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    X = GRU(units = 64, return_sequences = True)(X) # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                                 # Batch normalization
    
    # Step 3: Second GRU Layer (≈4 lines)
    X = GRU(units = 64, return_sequences = True)(X)   # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                                  # Batch normalization
    X = Dropout(0.8)(X)                                  # dropout (use 0.8)
    
    # Step 4: Time-distributed dense layer (≈1 line)
    X = Dense(1, activation = "relu")(X) # time distributed
    #X = TimeDistributed(Dense(1),activation="sigmoid")(X) # time distributed  (sigmoid)
    
    model = Model(inputs = X_input, outputs = X)
    
    return model

#sauce: https://github.com/Gurupradeep/deeplearning.ai-Assignments/blob/master/Sequence%20Models/Week3/Trigger%2Bword%2Bdetection%2B-%2Bv1.ipynb

#initialize the model, and compile it with its hyperparameters.
model = model(input_shape = (None,12))
opt = Adam(lr=0.0006)
model.compile(loss='mae', optimizer=opt, metrics=["mae"])

his = []
c=0

flatten = lambda l: [item for sublist in l for item in sublist]

filepath="model_checkpointfile.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=1, save_best_only=False, mode='min')
callbacks_list = [checkpoint]

history = model.fit_generator(train_gen, steps_per_epoch = 1000, epochs =15, callbacks=callbacks_list,validation_data=valid_gen, validation_steps=200) 
his.append(history.history['val_mean_absolute_error'])

his_flattened = flatten(his)

plt.subplots(figsize=(16,9))
plt.subplot(1,1,1)
plt.plot(his_flattened)

