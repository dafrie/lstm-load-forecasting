# module imports
import os
import sys
import math
from decimal import *
import itertools
import time as t
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas import datetime
from numpy import newaxis
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.tsa import stattools

import math
import keras as keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
from keras.callbacks import TensorBoard, EarlyStopping
from keras.utils import np_utils

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tabulate import tabulate


def generate_combinations(model_name=None, layer_conf=None, cells=None, dropout=None, batch_size=None, timesteps=None):
    """ Generates from parameters all possible combinations
    
    ...
    
    """
    # If also the dropout should have multiple settings per layer, recursion would be needed. But lets remain simple for now...
    # Creating the permutaions (possible combinations) between the different configuration parameters
    models = []
    
    layer_comb = list(itertools.product(*cells))
    configs = [layer_comb, dropout, batch_size, timesteps]
    combinations = list(itertools.product(*configs))
    
    for ix, comb in enumerate(combinations):
        m_name = model_name
        m_name += str(ix+1)
        
        # Now the list of layers needs to be generated
        layers = []
        for idx, level in enumerate(comb[0]):
            
            return_sequence = True
            # Inner loop, to check if all later layers have cell-sizes of 0.
            if all(size == 0 for size in comb[0][idx+1:]) == True:
                return_sequence = False
            if (idx+1) == len(comb[0]):
                return_sequence = False
            if level > 0:
                layers.append({'type': 'lstm', 'cells': level, 'dropout': comb[1], 
                               'stateful': layer_conf[idx], 'ret_seq': return_sequence }) 
                m_name += '_l-' + str(level)
                
        # Add dropout identifier to name
        if comb[1] > 0:
            m_name += '_d-' + str(comb[1])
        # Add model config
        model_config = {
            'name': m_name, 
            'layers': layers, 
            'batch_size': comb[2], 
            'timesteps': comb[3]
        }
        models.append(model_config)
    
    print('==================================')
    print(tabulate([
        ['Number of model configs generated', len(combinations)]], 
        tablefmt="jira", numalign="right", floatfmt=".3f"))
    
    return models
                          
              
def create_model(layers=None, sample_size=None, batch_size=1, timesteps=1, features=None, loss='mse', optimizer='adam'):
    """ Initializes a LSTM NN.
    
    ...
    
    Parameters
    ----------
    lstm_layers : dict
        Sets up the LSTM layers: [{'type': 'lstm, 'cells': 50, 'dropout': 0, 'ret_seq': False, 'stateful': True }]
    input_shape : tuple
        Needs in following form: (sample_size, timesteps, features)
    batch_input_shape : tuple
        Needs in following form: (batch_size, timesteps, features)
    loss : str
    
    optimizer : str
    
    Returns
    -------
    model
         Keras Model
    """
    
    model = Sequential()
    # For all configured layers
    
    for idx, l in enumerate(layers):
        if idx == 0:
            model.add(LSTM(l["cells"],
                           input_shape=(sample_size,timesteps,features),
                           batch_input_shape=(batch_size,timesteps,features),
                           return_sequences=l["ret_seq"],
                           stateful=l["stateful"],# TODO: Could actually set to True always
                          )
                     )
        # Only add additional layers, if cell is > 0
        elif idx > 0 and l["cells"] > 0:
            model.add(LSTM(l["cells"], return_sequences=l["ret_seq"], stateful=l["stateful"]))         
        # Add dropout.
        if l['dropout'] > 0:
            model.add(Dropout(l['dropout']))
    model.add(Dense(1))
    #model.add(Activation('tanh'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
    
    return model


def train_model(model=None, y=None, X=None, mode='train_on_batch', batch_size=1, timesteps=1, 
                epochs=1, rearrange=False, verbose=0, validation_split=0.1, early_stopping=True, min_delta=0.006, patience=2):
    """ Trains the model
    
    ...
    
    Parameters
    ----------
    mode : str
              Either 'train_on_batch' or 'fit'.
    Returns
    -------
    """
    # Set clock
    start_time = t.time()
    stopper = t.time()
    
    # Set callbacks
    # TODO: Add Keras callback such that log files will be written for TensorBoard integration
    # tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, 
    # write_graph=True,   write_images=True)
    # Set early stopping (check if model has converged) callbacks
    if early_stopping:
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, 
                                                   verbose=verbose, mode='auto')
    else:
        early_stop = keras.callbacks.Callback()
    
    if mode == 'train_on_batch':
        history = {}
        history['loss'] = []
        history['accu'] = []
        for epoch in range(epochs):
            
            mean_train_loss = []
            mean_train_accu = []
            num_batches = int(X.shape[0] / batch_size)
            for s in range(num_batches): 
                loss, accu = model.train_on_batch(
                    np.reshape(np.array(X[s*batch_size:(s+1)*batch_size,:]), (batch_size,timesteps,X.shape[1])),
                    y[s*batch_size:(s+1)*batch_size], callbacks=[early_stop]
                )
                mean_train_loss.append(loss)
                mean_train_accu.append(accu)
            # TODO: Check if here or above makes more sense.
            model.reset_states()
            
            if verbose > 0:
                print('============ Epoch {}/{}: State reset ============='.format(epoch + 1, epochs))
                print(tabulate([['Epoch MAE', np.mean(mean_train_accu)], 
                                ['Epoch Training loss', np.mean(mean_train_loss)], 
                                ['Train duration (s)', t.time() - start_time],
                                ['Since last epoch (s)', t.time() - stopper]
                               ], tablefmt="jira", numalign="right", floatfmt=".3f"))
            stopper = t.time()
            history['loss'].append(mean_train_loss)
            history['acc'].append(mean_train_accu)
            
    elif mode == 'fit':
       
        max_batch_count = int(X.shape[0] / batch_size)
        train_quotient = ((1-validation_split) * X.shape[0]) / batch_size
        valid_quotient = (X.shape[0] * validation_split) / batch_size
        train_count = int(train_quotient)
        valid_count = int(valid_quotient)
        
        # If a whole batch got "lost" by double rounding, then lets add it back to the one which was closer to a "full" batch.
        if train_count + valid_count < max_batch_count:
            if (train_quotient - train_count) > (valid_quotient - valid_count):
                train_count += 1
            else:
                valid_count += 1
        
        effective_split = valid_count / max_batch_count
        
        X_train = X[0:max_batch_count*batch_size]
        y_train = y[0:max_batch_count*batch_size]
            
        if (max_batch_count * batch_size) < (X.shape[0]) and verbose > 0:
            print('Warnining: Division "sample_size/batch_size" not a natural number.')
            print('Dropped the last {} of {} number of obs.'.format(X.shape[0] - max_batch_count*batch_size, X.shape[0]))
            print('Effective validation split now is: {0:0.3f}'.format(effective_split))

        # Fitting the model. Keras function
        history = model.fit(
            shuffle=False,
            x=np.reshape(np.array(X_train), (X_train.shape[0], timesteps, X_train.shape[1])),
            y=y_train,
            validation_split=effective_split,
            batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=[early_stop]
        )
        
    else:
        raise ValueException("The mode '{}' was invalid. Must be either fit or train_on_batch".format(mode))
    
    return history
    
    
def plot_history(model_config=None, history=None, path=None, metrics='mean_absolute_error', interactive=False, display=False):
    # Turn interactive plotting off
    if not interactive:
        plt.ioff()
        
    # summarize history for accuracy
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Training history: Mean absolute error ')
    plt.ylabel('Mean absolute error')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    
    filename = path + model_config['name'] + '_mae'
    plt.savefig(filename + '.pgf')
    plt.savefig(filename + '.pdf')
    
    if display:
        plt.show()
    elif not display:
        # Reset
        plt.clf()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training history: Loss ')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    
    filename = path + model_config['name'] + '_loss'
    plt.savefig(filename + '.pgf')
    plt.savefig(filename + '.pdf')
    
    if display:
        plt.show()
    elif not display:
        # Reset
        plt.clf()
        
    
def evaluate_model(model=None, X=None, y=None, batch_size=1, timesteps=None, verbose=0):
    
    max_batch_count = int(X.shape[0] / batch_size)
    
    if (max_batch_count * batch_size) < (X.shape[0]) and verbose > 0:
        print('Warnining: Division "sample_size/batch_size" not a natural number.')
        print('Dropped the last {} of {} number of obs.'.format(X.shape[0] - max_batch_count*batch_size, X.shape[0]))
    
    X = X[0:max_batch_count*batch_size]
    y = y[0:max_batch_count*batch_size]
    X = np.reshape(np.array(X), (X.shape[0], timesteps, X.shape[1]))
    test_loss, test_mae = model.evaluate(X, y, batch_size=batch_size, verbose=verbose, sample_weight=None)
    
    return test_loss, test_mae


def get_predictions(model=None, X=None, batch_size=1, timesteps=1, verbose=0):
        
    max_batch_count = int(X.shape[0] / batch_size)
    
    if (max_batch_count * batch_size) < (X.shape[0]) and verbose > 0:
        print('Warnining: Division "sample_size/batch_size" not a natural number.')
        print('Dropped the last {} of {} number of obs.'.format(X.shape[0] - max_batch_count*batch_size, X.shape[0]))
    
    X = X[0:max_batch_count*batch_size]
    X = np.reshape(np.array(X), (X.shape[0], timesteps, X.shape[1]))
    predictions = model.predict(x=X, batch_size=batch_size, verbose=verbose)
    
    return predictions


def plot_predictions(predictions=None):
        # Turn interactive plotting off
    if not interactive:
        plt.ioff()
        
    # summarize history for accuracy
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Training history: Mean absolute error ')
    plt.ylabel('Mean absolute error')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    
    filename = path + model_config['name'] + '_mae'
    plt.savefig(filename + '.pgf')
    plt.savefig(filename + '.pdf')
    
    if display:
        plt.show()
    elif not display:
        # Reset
        plt.clf()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training history: Loss ')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    
    filename = path + model_config['name'] + '_loss'
    plt.savefig(filename + '.pgf')
    plt.savefig(filename + '.pdf')
    
    if display:
        plt.show()
    elif not display:
        # Reset
        plt.clf()

