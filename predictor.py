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
from tensorflow.keras.models import load_model
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
training = False
oneFeature = True
if training == False:
    if oneFeature:
        regressor = load_model("modeloneinput.h5")
    input_feature = dataset_train[['Close']]
    input_data = input_feature.values
    print("type", type(input_data))
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # input_data[:,:] = scaler.fit_transform(input_data[:,:])

    # input_data = scaler.fit_transform(input_data)
    # Normalize the data
    inputnew = []
    print("going to scale")
    print("shape input", input_data.shape)
    katt = 0
    rangestuff = 8
    print(len(input_data))
    a = np.zeros((len(input_data), 1))
    for i in range(rangestuff):
        scaler = MinMaxScaler(feature_range=(0, 1))
        # input_data[:,:] = scaler.fit_transform(input_data[:,:])

        t = scaler.fit_transform(
            input_data[int(i * len(input_data) / rangestuff):int((i + 1) * len(input_data) / rangestuff), :])
        a[int(i * len(input_data) / rangestuff):int((i + 1) * len(input_data) / rangestuff), :] = t

        inputnew.append(t)

    input_data = a

    X = []
    y = []
    lookback = 30
    total_size = len(dataset_train)
    print("shape input after", input_data.shape)
    for i in range(0, total_size - lookback):  # loop data set with margin 50 as we use 50 days data for prediction
        t = []
        for j in range(0, lookback):  # loop for 50 days
            current_index = i + j
            t.append(input_data[current_index, :])  # get data margin from 50 days with marging i

        # X 12580x 30x 5
        # t 30 x 5

        X.append(t)

        y.append(input_data[lookback + i, 0])

    X, y = np.array(X), np.array(y)
    print("shapes x_train and y_train")
    print(X.shape, y.shape)

    test_size = 600

    X_test = X[-test_size:]
    Y_test = y[-test_size:]

    X_rest = X[: -test_size]
    y_rest = y[: -test_size]

    X_train, X_valid, y_train, y_valid = train_test_split(X_rest, y_rest, test_size=0.15, random_state=101)
    predictedvalues = []
    x_t = X_test[0]

    print("shape" , X_test.shape)
    x_t = x_t.reshape(1, lookback, 1)
    print("shape", x_t.shape)
    startp = 0
    predictlen = 20
    X_test = X_test[startp:startp+predictlen]
    Y_test = Y_test[startp:startp+predictlen]
    for i in range(predictlen):
        predicted_value = regressor.predict(x_t)
        predictedvalues.append(predicted_value[0])
        x_new = x_t
        x_new[0][lookback-1] = predicted_value
        x_new[0][0:lookback-1] = x_t[0][1:]
    print(predictedvalues)
    plt.figure(figsize=(18, 8))
    plt.plot(predictedvalues, label='Predicted Value')
    plt.plot(Y_test, label='Real Value')
    plt.legend(loc='upper left')
    plt.title("Close Value of S&P 500")
    plt.xlabel("Days")
    plt.ylabel("Close Value")
    plt.show()





if training ==True:










    if oneFeature:


        #keras only takes numpy array

        #scaler = MinMaxScaler(feature_range = (0, 1))
        #fit: get min/max of train data
        #training_set_scaled = scaler.fit_transform(training_set)
        input_feature = dataset_train[['Close']]
        input_data = input_feature.values
        print("type", type(input_data))
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # input_data[:,:] = scaler.fit_transform(input_data[:,:])

        # input_data = scaler.fit_transform(input_data)
        # Normalize the data
        inputnew = []
        print("going to scale")
        print("shape input", input_data.shape)
        katt = 0
        rangestuff = 8
        print(len(input_data))
        a = np.zeros((len(input_data), 1))
        for i in range(rangestuff):
            scaler = MinMaxScaler(feature_range=(0, 1))
            # input_data[:,:] = scaler.fit_transform(input_data[:,:])

            t = scaler.fit_transform(
                input_data[int(i * len(input_data) / rangestuff):int((i + 1) * len(input_data) / rangestuff), :])
            a[int(i * len(input_data) / rangestuff):int((i + 1) * len(input_data) / rangestuff), :] = t

            inputnew.append(t)

        input_data = a

        X = []
        y = []
        lookback =30
        total_size = len(dataset_train)
        print("shape input after", input_data.shape)
        for i in range(0, total_size - lookback):  # loop data set with margin 50 as we use 50 days data for prediction
            t = []
            for j in range(0, lookback):  # loop for 50 days
                current_index = i + j
                t.append(input_data[current_index, :])  # get data margin from 50 days with marging i

            # X 12580x 30x 5
            # t 30 x 5

            X.append(t)

            y.append(input_data[lookback + i, 0])

        X, y = np.array(X), np.array(y)
        print("shapes x_train and y_train")
        print(X.shape, y.shape)

        test_size = 600

        X_test = X[-test_size:]
        Y_test = y[-test_size:]

        X_rest = X[: -test_size]
        y_rest = y[: -test_size]

        X_train, X_valid, y_train, y_valid = train_test_split(X_rest, y_rest, test_size=0.15, random_state=101)


        regressor = Sequential()
        # add 1st lstm layer
        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        regressor.add(Dropout(rate=0.2))

        ##add 2nd lstm layer: 50 neurons
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(rate=0.2))

        ##add 3rd lstm layer
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(rate=0.2))

        ##add 4th lstm layer
        regressor.add(LSTM(units=50, return_sequences=False))
        regressor.add(Dropout(rate=0.2))

        ##add output layer
        regressor.add(Dense(units=1))

        regressor.compile(optimizer='adam', loss='mean_squared_error')
        regressor.fit(x=X_train, y=y_train, batch_size=32, epochs=100)


        predicted_value = regressor.predict(X_test)
        regressor.save("modeljkdnsjkfsdjnk.h5")
        plt.figure(figsize=(18, 8))
        plt.plot(predicted_value, label = 'Predicted Value')
        plt.plot(Y_test, label = 'Real Value')
        plt.legend(loc='upper left')
        plt.title("Close Value of S&P 500")
        plt.xlabel("Days")
        plt.ylabel("Close Value")
        plt.show()




    else:
        # Selecting 5 interesting feature
        input_feature = dataset_train[['Open', 'High', 'Low', 'Volume', 'Close']]
        input_data = input_feature.values
        print("type", type(input_data))
        #scaler = MinMaxScaler(feature_range=(0, 1))
        # input_data[:,:] = scaler.fit_transform(input_data[:,:])

        #input_data = scaler.fit_transform(input_data)
        # Normalize the data
        inputnew = []
        print("going to scale")
        print("shape input" , input_data.shape)
        katt = 0
        rangestuff =8
        a = np.zeros((int((rangestuff+1)*len(input_data)/rangestuff),5))
        for i in range(rangestuff):

            scaler = MinMaxScaler(feature_range=(0,1))
            # input_data[:,:] = scaler.fit_transform(input_data[:,:])

            t = scaler.fit_transform(input_data[int(i*len(input_data)/rangestuff):int((i+1)*len(input_data)/rangestuff),:])
            a[int(i*len(input_data)/rangestuff):int((i+1)*len(input_data)/rangestuff),:] = t

            inputnew.append(t)

        input_data = a



        lookback = 30
        total_size = len(dataset_train)

        X=[]
        y=[]
        print("shape input after" , input_data.shape)
        for i in range(0, total_size-lookback): # loop data set with margin 50 as we use 50 days data for prediction
            t=[]
            for j in range(0, lookback): # loop for 50 days
                current_index = i+j
                t.append(input_data[current_index, :]) # get data margin from 50 days with marging i

            # X 12580x 30x 5
            # t 30 x 5

            X.append(t)


            y.append(input_data[lookback+i, 4])

        X, y= np.array(X), np.array(y)
        print("shapes x_train and y_train")
        print(X.shape, y.shape)

        test_size = 600

        X_test = X[-test_size:]
        Y_test = y[-test_size:]

        X_rest = X[: -test_size]
        y_rest = y[: -test_size]

        X_train, X_valid, y_train, y_valid = train_test_split(X_rest, y_rest, test_size = 0.15, random_state = 101)

        X_train = X_train.reshape(X_train.shape[0], lookback, 1)
        X_valid = X_valid.reshape(X_valid.shape[0], lookback, 1)
        X_test = X_test.reshape(X_test.shape[0], lookback, 1)


        print("shapes x_train shape x_valid, X-test")
        print(X_train.shape)
        print(X_valid.shape)
        print(X_test.shape)
        print(X_train[1][:][:])
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
        regressor.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_valid, y_valid), callbacks=callbacks)
        predicted_value = regressor.predict(X_test)

        plt.figure(figsize=(18, 8))
        plt.plot(predicted_value, label = 'Predicted Value')
        plt.plot(Y_test, label = 'Real Value')
        plt.legend(loc='upper left')
        plt.title("Close Value of S&P 500")
        plt.xlabel("Days")
        plt.ylabel("Close Value")
        plt.show()

