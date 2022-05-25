#!pip install pandas
#!pip install pandas_datareader
#!pip install plotly
#!pip install tensorflow==2.8
#!pip install yfinance
#!pip install yahoofinancials
#!pip install keras

import time
import datetime as dt
import pandas as pd
from pandas_datareader import data

import plotly.graph_objs as go
import matplotlib.pyplot as plt
import urllib.request, json
import os
import numpy as np
import tensorflow.compat.v1 as tf # This code has been tested with TensorFlow 1
tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()
from sklearn.preprocessing import MinMaxScaler

# Yahoo tools for downloading Yahoo market data
import yfinance as yf
from yahoofinancials import YahooFinancials


def PlotCandelstick(stock):
    data = yf.download(tickers=stock, period = "5y", interval = "1d", rounding= True)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,open = data["Open"], high=data["High"], low=data["Low"], close=data["Close"], name = "market data"))
    fig.update_layout(title = stock + "share price", yaxis_title = "Stock Price (USD)")
    fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
    buttons=list([
    dict(count=15, label="15m", step="minute", stepmode="backward"),
    dict(count=45, label="45m", step="minute", stepmode="backward"),
    dict(count=1, label="1h", step="hour", stepmode="backward"),
    dict(count=6, label="6h", step="hour", stepmode="backward"),
    dict(step="all")
    ])
    )
    )
    fig.show()


def splitdataClosed(dataframe):
    # The training data will be the first 80% of data points and the rest will be test data.
    close = dataframe['Close'].to_numpy()
    size = close.size

    # The training data will be the first 80% of data points and the rest will be test data.
    train_test_fraction = 0.8
    train_data = close[:(round(size*train_test_fraction))]
    validation_data = close[(round(size*train_test_fraction)):]
    return train_data, validation_data


def splitdata(dataframe):
    size = dataframe['High'].size

    # The training data will be the first 80% of data points and the rest will be test data.
    train_test_fraction = 0.8
    train_data = dataframe.iloc[0:(round(size*train_test_fraction)), 1:4].to_numpy()
    validation_data = dataframe.iloc[(round(size*train_test_fraction)):,1:4].to_numpy()

    return train_data, validation_data


def averageData(dataframe):
    high = stock_df['High'].to_numpy()
    low = stock_df['Low'].to_numpy()
    average = (high+low)/2.0
    size = average.size

    # The training data will be the first 80% of data points and the rest will be test data.
    train_test_fraction = 0.8
    train_data = average[:(round(size*train_test_fraction))]
    validation_data = average[(round(size*train_test_fraction)):]

    return train_data, validation_data



def normalize(train_data, validation_data):
    scaler = MinMaxScaler()
    print(train_data.shape)
    train_data = train_data.reshape(-1,1)
    validation_data = validation_data.reshape(-1,1)
    print("hello")
    print(train_data.shape)

    # Train the Scaler with training data and smooth data
    smoothing_window_size = 500
    for di in range(0, 2000, smoothing_window_size):
        scaler.fit(train_data[di:di + smoothing_window_size, :])
        train_data[di:di + smoothing_window_size, :] = scaler.transform(train_data[di:di + smoothing_window_size, :])

    # You normalize the last bit of remaining data
    scaler.fit(train_data[di + smoothing_window_size:, :])
    train_data[di + smoothing_window_size:, :] = scaler.transform(train_data[di + smoothing_window_size:, :])
    # Reshape both train and test data
    train_data = train_data.reshape(-1)

    # Normalize test data
    validation_data = scaler.transform(validation_data).reshape(-1)
    print("hello2")
    print(train_data.shape)
    print(train_data)
    return train_data, validation_data

def smoothdata(train_data,test_data ):
    # Now perform exponential moving average smoothing
    # So the data will have a smoother curve than the original ragged data
    EMA = 0.0
    gamma = 0.1

    for ti in range(train_data.size):
        EMA = gamma * train_data[ti] + (1 - gamma) * EMA
        train_data[ti] = EMA

    # Used for visualization and test purposes
    all_mid_data = np.concatenate([train_data, test_data], axis=0)
    return train_data, all_mid_data

# LSTMs: Making Stock Predictions Far into the Future
class DataGeneratorSeq(object):

    def __init__(self, prices, batch_size, num_unroll):
        self._prices = prices
        self._prices_length = len(self._prices) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._prices_length //self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):

        batch_data = np.zeros((self._batch_size), dtype=np.float32)
        batch_labels = np.zeros((self._batch_size), dtype=np.float32)

        for b in range(self._batch_size):
            if self._cursor[b]+1 >= self._prices_length:
                #self._cursor[b] = b * self._segments
                self._cursor[b] = np.random.randint(0, (b+1) * self._segments)

            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b] = self._prices[self._cursor[b] + np.random.randint(0,5)]

            self._cursor[b] = (self._cursor[b]+1)%self._prices_length

        return batch_data, batch_labels

    def unroll_batches(self):

        unroll_data,unroll_labels = [],[]
        init_data, init_label = None,None
        for ui in range(self._num_unroll):

            data, labels = self.next_batch()

            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0,min((b + 1) * self._segments, self._prices_length - 1))





"""
plt.figure(figsize = (18,9))
plt.plot(range(stock_df.shape[0]),(stock_df["Low"]+stock_df["High"])/2.0)
plt.xticks(range(0,stock_df.shape[0],500),stock_df.index[::500],rotation=45)
plt.xlabel("Date",fontsize=18)
plt.ylabel("Mid Price",fontsize=18)
plt.show()
"""


# Marknadsindex - S&P 500 ("^GSPC")
# Apples stock - ("AAPL")

#stock_names = ["^GSPC", "AAL", "AAPL"] #"AAPL"
stock_names = ["^GSPC"]
all_train_data =[]
all_validation_data =[]

for stock in stock_names:
    print("Stock name")

    gspc = yf.Ticker(stock)
    stock_df = gspc.history(period="50y")
    stock_df.index.name = "Date"
    stock_df = stock_df.sort_values('Date')

    print("stock_df.shape: ", stock_df.shape)

    #print(stock_df.head()) # Jag har verifierat att nedladdad data stämmer överens med Yahoo finance
    train_data, validation_data = splitdataClosed(stock_df)
    print("train_data.shape: ", train_data.shape)
    print("validation_data.shape: ", validation_data.shape)

    train_data, validation_data = normalize(train_data, validation_data)
    train_data, all_mid_data = smoothdata(train_data, validation_data)




    print("all_mid_data.shape: ", all_mid_data.shape)


    all_train_data.append(train_data)
    all_validation_data.append(validation_data)



    # LSTM test
    batch_size = 500 # Number of samples in a batch

    dg = DataGeneratorSeq(train_data, 5, 5)
    u_data, u_labels = dg.unroll_batches()

    for ui,(dat,lbl) in enumerate(zip(u_data, u_labels)):
        print('\n\nUnrolled index %d'%ui)
        dat_ind = dat
        lbl_ind = lbl
        print('\tInputs: ',dat )
        print('\n\tOutput:',lbl)


    # ======================== LSTM training ===================================================
    # Hyperparameters
    print("train_data.shape =: ", train_data.shape)
    D = 1 # Dimensionality of the data. Since your data is 1-D this would be 1
    num_unrollings = 50 # Number of time steps you look into the future.
    num_nodes = [500,500,100] # Number of hidden nodes in each layer of the deep LSTM stack we're using
    n_layers = len(num_nodes) # number of layers
    dropout = 0.2 # dropout amount
    #tf.reset_default_graph() # This is important in case we run multiple times

    # Input data.
    train_inputs, train_outputs = [],[]

    # You unroll the input over time defining placeholders for each time step
    for ui in range(num_unrollings):
        train_inputs.append(tf.compat.v1.placeholder(tf.float32, shape=[batch_size,D],name='train_inputs_%d'%ui))
        train_outputs.append(tf.compat.v1.placeholder(tf.float32, shape=[batch_size,1], name = 'train_outputs_%d'%ui))

    lstm_cells = [
        tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=num_nodes[li],
                                state_is_tuple=True,
                                initializer= tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
                               )
     for li in range(n_layers)]

    drop_lstm_cells = [tf.compat.v1.nn.rnn_cell.DropoutWrapper(
        lstm, input_keep_prob=1.0,output_keep_prob=1.0-dropout, state_keep_prob=1.0-dropout
    ) for lstm in lstm_cells]
    drop_multi_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(drop_lstm_cells)
    multi_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(lstm_cells)

    w = tf.compat.v1.get_variable('w',shape=[num_nodes[-1], 1], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
    b = tf.compat.v1.get_variable('b',initializer=tf.random.uniform([1],-0.1,0.1))

    # Create cell state and hidden state variables to maintain the state of the LSTM
    c, h = [],[]
    initial_state = []
    for li in range(n_layers):
      c.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
      h.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
      initial_state.append(tf.nn.rnn_cell.LSTMStateTuple(c[li], h[li]))

    # Do several tensor transofmations, because the function dynamic_rnn requires the output to be of
    # a specific format. Read more at: https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
    all_inputs = tf.concat([tf.expand_dims(t,0) for t in train_inputs],axis=0)

    # all_outputs is [seq_length, batch_size, num_nodes]
    all_lstm_outputs, state = tf.compat.v1.nn.dynamic_rnn(
        drop_multi_cell, all_inputs, initial_state=tuple(initial_state),
        time_major = True, dtype=tf.float32)

    all_lstm_outputs = tf.reshape(all_lstm_outputs, [batch_size*num_unrollings,num_nodes[-1]])

    all_outputs = tf.compat.v1.nn.xw_plus_b(all_lstm_outputs,w,b)

    split_outputs = tf.split(all_outputs,num_unrollings,axis=0)

    # When calculating the loss you need to be careful about the exact form, because you calculate
    # loss of all the unrolled steps at the same time
    # Therefore, take the mean error or each batch and get the sum of that over all the unrolled steps

    print('Defining training Loss')
    loss = 0.0
    with tf.control_dependencies([tf.compat.v1.assign(c[li], state[li][0]) for li in range(n_layers)]+
                                 [tf.compat.v1.assign(h[li], state[li][1]) for li in range(n_layers)]):
      for ui in range(num_unrollings):
        loss += tf.reduce_mean(input_tensor=0.5*(split_outputs[ui]-train_outputs[ui])**2)

    print('Learning rate decay operations')
    global_step = tf.Variable(0, trainable=False)
    inc_gstep = tf.compat.v1.assign(global_step,global_step + 1)
    tf_learning_rate = tf.compat.v1.placeholder(shape=None,dtype=tf.float32)
    tf_min_learning_rate = tf.compat.v1.placeholder(shape=None,dtype=tf.float32)

    learning_rate = tf.maximum(
        tf.compat.v1.train.exponential_decay(tf_learning_rate, global_step, decay_steps=1, decay_rate=0.5, staircase=True),
        tf_min_learning_rate)

    # Optimizer.
    print('TF Optimization operations')
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer = optimizer.apply_gradients(
        zip(gradients, v))

    print('\tAll done')


    print('Defining prediction related TF functions')

    sample_inputs = tf.compat.v1.placeholder(tf.float32, shape=[1,D])

    # Maintaining LSTM state for prediction stage
    sample_c, sample_h, initial_sample_state = [],[],[]
    for li in range(n_layers):
      sample_c.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
      sample_h.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
      initial_sample_state.append(tf.nn.rnn_cell.LSTMStateTuple(sample_c[li],sample_h[li]))

    reset_sample_states = tf.group(*[tf.compat.v1.assign(sample_c[li],tf.zeros([1, num_nodes[li]])) for li in range(n_layers)],
                                   *[tf.compat.v1.assign(sample_h[li],tf.zeros([1, num_nodes[li]])) for li in range(n_layers)])

    sample_outputs, sample_state = tf.compat.v1.nn.dynamic_rnn(multi_cell, tf.expand_dims(sample_inputs,0),
                                       initial_state=tuple(initial_sample_state),
                                       time_major = True,
                                       dtype=tf.float32)

    with tf.control_dependencies([tf.compat.v1.assign(sample_c[li],sample_state[li][0]) for li in range(n_layers)]+
                                  [tf.compat.v1.assign(sample_h[li],sample_state[li][1]) for li in range(n_layers)]):
      sample_prediction = tf.compat.v1.nn.xw_plus_b(tf.reshape(sample_outputs,[1,-1]), w, b)

    print('\tAll done')


    #  ========================= Running the LSTM =================================
    epochs = 10
    valid_summary = 1 # Interval you make test predictions

    n_predict_once = 1 # Number of steps you continously predict for

    train_seq_length = train_data.size # Full length of the training data
    print("Train_data.size = ", train_data.size)

    train_mse_ot = [] # Accumulate Train losses
    test_mse_ot = [] # Accumulate Test loss
    predictions_over_time = [] # Accumulate predictions

    session = tf.compat.v1.InteractiveSession()

    tf.compat.v1.global_variables_initializer().run()

    # Used for decaying learning rate
    loss_nondecrease_count = 0
    loss_nondecrease_threshold = 2 # If the test error hasn't increased in this many steps, decrease learning rate

    print('Initialized')
    average_loss = 0

    # Define data generator
    data_gen = DataGeneratorSeq(train_data,batch_size,num_unrollings)

    x_axis_seq = []

    # Points you start your test predictions from
    test_points_seq = np.arange(train_data.size,train_data.size+1000,50).tolist()
    #test_points_seq = np.arange(train_data.size*0.8,train_data.size,50).tolist()


    for ep in range(epochs):

        # ========================= Training =====================================
        for step in range(train_seq_length//batch_size):

            u_data, u_labels = data_gen.unroll_batches()

            feed_dict = {}
            for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):
                feed_dict[train_inputs[ui]] = dat.reshape(-1,1)
                feed_dict[train_outputs[ui]] = lbl.reshape(-1,1)

            feed_dict.update({tf_learning_rate: 0.0001, tf_min_learning_rate:0.000001})

            _, l = session.run([optimizer, loss], feed_dict=feed_dict)

            average_loss += l

        # ============================ Validation ==============================
        if (ep+1) % valid_summary == 0:

          average_loss = average_loss/(valid_summary*(train_seq_length//batch_size))

          print('Average loss at step %d: %f' % (ep+1, average_loss))

          train_mse_ot.append(average_loss)

          average_loss = 0 # reset loss

          predictions_seq = []

          mse_test_loss_seq = []

          # ===================== Updating State and Making Predicitons ========================
          for w_i in test_points_seq:
            mse_test_loss = 0.0
            our_predictions = []

            if (ep+1)-valid_summary==0:
              # Only calculate x_axis values in the first validation epoch
              x_axis=[]

            # Feed in the recent past behavior of stock prices
            # to make predictions from that point onwards
            for tr_i in range(w_i-num_unrollings+1,w_i-1):
              current_price = all_mid_data[tr_i]
              feed_dict[sample_inputs] = np.array(current_price).reshape(1,1)
              _ = session.run(sample_prediction,feed_dict=feed_dict)

            feed_dict = {}

            current_price = all_mid_data[w_i-1]

            feed_dict[sample_inputs] = np.array(current_price).reshape(1,1)

            # Make predictions for this many steps
            # Each prediction uses previous prediciton as it's current input
            for pred_i in range(n_predict_once):

              pred = session.run(sample_prediction,feed_dict=feed_dict)

              our_predictions.append(np.asscalar(pred))

              feed_dict[sample_inputs] = np.asarray(pred).reshape(-1,1)

              if (ep+1)-valid_summary==0:
                # Only calculate x_axis values in the first validation epoch
                x_axis.append(w_i+pred_i)

              mse_test_loss += 0.5*(pred-all_mid_data[w_i+pred_i])**2

            session.run(reset_sample_states)

            predictions_seq.append(np.array(our_predictions))

            mse_test_loss /= n_predict_once
            mse_test_loss_seq.append(mse_test_loss)

            if (ep+1)-valid_summary==0:
              x_axis_seq.append(x_axis)

          current_test_mse = np.mean(mse_test_loss_seq)

          # Learning rate decay logic
          if len(test_mse_ot)>0 and current_test_mse > min(test_mse_ot):
              loss_nondecrease_count += 1
          else:
              loss_nondecrease_count = 0

          if loss_nondecrease_count > loss_nondecrease_threshold :
                session.run(inc_gstep)
                loss_nondecrease_count = 0
                print('\tDecreasing learning rate by 0.5')

          test_mse_ot.append(current_test_mse)
          print('\tTest MSE: %.5f'%np.mean(mse_test_loss_seq))
          predictions_over_time.append(predictions_seq)
          print('\tFinished Predictions')


    # Plot training and test loss
    plt.figure()
    plt.plot(range(0, epochs), train_mse_ot, label="train loss")
    plt.plot(range(0, epochs), test_mse_ot, label="test loss")
    plt.legend(loc='upper right')

    plt.title('Training and test loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.show()


    # Plot the predicitons
    val = epochs
    #val = input("Enter your epoch with best results: ")
    best_prediction_epoch = val-1 # replace this with the epoch that you got the best results when running the plotting code

    plt.figure(figsize = (18,18))
    plt.subplot(2,1,1)
    plt.plot(range(stock_df.shape[0]), all_mid_data, color='b')

    # Plotting how the predictions change over time
    # Plot older predictions with low alpha and newer predictions with high alpha
    start_alpha = 0.25
    alpha  = np.arange(start_alpha,1.1,(1.0-start_alpha)/len(predictions_over_time[::3]))
    for p_i,p in enumerate(predictions_over_time[::3]):
        for xval,yval in zip(x_axis_seq,p):
            plt.plot(xval,yval,color='r',alpha=alpha[p_i])

    plt.title('Different Predictions Over Time',fontsize=18)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Average Price',fontsize=18)
    plt.xlim(train_data.size,train_data.size+1500)

    plt.subplot(2,1,2)

    # Predicting the best test prediction you got
    plt.plot(range(stock_df.shape[0]),all_mid_data,color='b')
    for xval,yval in zip(x_axis_seq,predictions_over_time[best_prediction_epoch]):
        plt.plot(xval,yval,color='r')

    plt.title('Best Test Predictions Over Time',fontsize=18)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Average Price',fontsize=18)
    plt.xlim(train_data.size,train_data.size+1500)
    plt.show()

#PlotCandelstick(stock_name)
#stock_df["Close"].plot(title= stock_name + "'s stock price")
#plt.show()
#print(stock_df.keys())
#print(stock_df.index[0])