#!pip install pandas
#!pip install pandas_datareader
#!pip install plotly
#!pip install tensorflow==2.8
#!pip install yfinance
#!pip install yahoofinancials
#!pip install keras

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

import os

# Yahoo tools for downloading Yahoo market data
import yfinance as yf
from yahoofinancials import YahooFinancials


def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


stock_names = ["^GSPC"]
all_train_data =[]
all_validation_data =[]

print("Stock name")

gspc = yf.Ticker(stock_names[0])
stock_df = gspc.history(period="50y")
stock_df.index.name = "Date"
stock_df = stock_df.sort_values('Date')
#stock_df['Close'] = stock_df['Close'].apply(lambda x: float(x.replace(',', '')))
#stock_df['Volume'] = stock_df['Volume'].apply(lambda x: float(x.replace(',', '')))

print("stock_df.shape: ", stock_df.shape)

dataset_train = stock_df

print(dataset_train.head())
print(dataset_train.sample(5))

plt.figure(figsize=(18, 8))
plt.plot(range(dataset_train.shape[0]), dataset_train['Close'], )
plt.title("S&P 500 Close Prices")
plt.xlabel("Days")
plt.ylabel("Close Price")
plt.show()


if os.path.exists('config.py'):
    print(1)
else:
    print(0)


oneFeature = False
if oneFeature:
    #keras only takes numpy array
    size = dataset_train['Open'].size

    training_set = dataset_train.iloc[:round(size*0.8), 1: 2].values

    scaler = MinMaxScaler(feature_range = (0, 1))
    #fit: get min/max of train data
    training_set_scaled = scaler.fit_transform(training_set)


    ## 60 timesteps and 1 output
    X_train = []
    y_train = []
    for i in range(60, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-60: i, 0])
        y_train.append(training_set_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    print(X_train.shape)
    print(y_train.shape)

    # Adding a 1 to indicate the dimension of the input (only looking at 'Open')
    # Number of stock prices - 1449
    # Number of time steps - 60
    # Number of Indicator - 1
    X_train = np.reshape(X_train, newshape =
                         (X_train.shape[0], X_train.shape[1], 1))

    print(X_train.shape)

    # Creating the architecture
    regressor = Sequential()
    #add 1st lstm layer: 512 neurons
    regressor.add(LSTM(units = 512, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(rate = 0.2))

    ##add 2nd lstm layer: 512 neurons
    regressor.add(LSTM(units = 512, return_sequences = True))
    regressor.add(Dropout(rate = 0.2))

    ##add 3rd lstm layer
    regressor.add(LSTM(units = 32, activation = 'relu', return_sequences = False))
    regressor.add(Dropout(rate = 0.2))

    ##add output layer
    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor.fit(x = X_train, y = y_train, batch_size = 512, epochs = 15)


    # The test data set
    real_stock_price = dataset_train.iloc[round(size*0.8):, 1: 2].values
    real_stock_price.shape

    dataset_total = dataset_train['Open']

    ##use .values to make numpy array
    inputs = dataset_total[len(training_set) - 60:].values


    #reshape data to only have 1 col
    inputs = inputs.reshape(-1, 1)
    #scale input
    inputs = scaler.transform(inputs)

    # Create test data strucutre
    X_test = []
    for i in range(60, len(inputs)):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    #add dimension of indicator
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    print("X_test.shape = ", X_test.shape)


    predicted_stock_price = regressor.predict(X_test)

    #inverse the scaled value
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)


    ##visualize the prediction and real price
    plt.plot(real_stock_price, label = 'Real price')
    plt.plot(predicted_stock_price, label = 'Predicted price')

    plt.title('Price prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


else:
    # Selecting 5 interesting feature
    input_feature = dataset_train[['Open', 'High', 'Low', 'Volume', 'Close']]
    input_data = input_feature.values

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0,1))
    # input_data[:,:] = scaler.fit_transform(input_data[:,:])
    input_data = scaler.fit_transform(input_data)

    lookback = 60
    total_size = len(dataset_train)

    X=[]
    y=[]
    for i in range(0, total_size-lookback): # loop data set with margin 50 as we use 50 days data for prediction
        t=[]
        for j in range(0, lookback): # loop for 50 days
            current_index = i+j
            t.append(input_data[current_index, :]) # get data margin from 50 days with marging i
        X.append(t)
        y.append(input_data[lookback+i, 4])

    X, y= np.array(X), np.array(y)
    print(X.shape, y.shape)

    test_size = 120

    X_test = X[-test_size:]
    Y_test = y[-test_size:]

    X_rest = X[: -test_size]
    y_rest = y[: -test_size]

    X_train, X_valid, y_train, y_valid = train_test_split(X_rest, y_rest, test_size = 0.15, random_state = 101)

    X_train = X_train.reshape(X_train.shape[0], lookback, 5)
    X_valid = X_valid.reshape(X_valid.shape[0], lookback, 5)
    X_test = X_test.reshape(X_test.shape[0], lookback, 5)
    print(X_train.shape)
    print(X_valid.shape)
    print(X_test.shape)

    # Creating the architecture
    regressor = Sequential()
    # add 1st layer
    regressor.add(LSTM(units = 512, return_sequences = True, input_shape = (X_train.shape[1], 5)))
    regressor.add(Dropout(rate = 0.2))

    # 2nd layer
    regressor.add(LSTM(units = 512, return_sequences = True))
    regressor.add(Dropout(rate = 0.2))

    # 3rd layer
    regressor.add(LSTM(units = 32, activation = 'relu', return_sequences = False))
    regressor.add(Dropout(rate = 0.2))

    # add output layer
    regressor.add(Dense(units = 1))

    # We introduc callbacks:
    # EarlyStoping: It will stop the traning if score of model didn't increase.
    # ReduceLROnPlateau: Use for reduce the learning rate.
    # ModelCheckpoint: Use for save model only when the score increased
    callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]

    # regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=[soft_acc])
    regressor.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_valid, y_valid), callbacks=callbacks)
    predicted_value = regressor.predict(X_test)

    plt.figure(figsize=(18, 8))
    plt.plot(predicted_value, label = 'Predicted price')
    plt.plot(Y_test, label = 'Real price')
    plt.title("Close price of stocks")
    plt.xlabel("Days")
    plt.ylabel("Stock Opening Price")
    plt.show()

