# Imports
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator, MONDAY  
from mplfinance.original_flavor import candlestick_ohlc
import pandas as pd
import numpy as np
from training import DQNAgent, preprocess

# Load data
csv_file_name = 'EURUSD_M30.csv'
data = pd.read_csv(csv_file_name, delimiter='\t')
data['Time'] = pd.to_datetime(data['Time'])
data.set_index('Time', inplace=True)

# Get last 50 historical candles  
hist = data.iloc[-100:]

# Make 60 predictions
state_size = 100
action_size = 3
weights_file = 'dqn_weights.h5'

agent = DQNAgent(state_size, action_size)
agent.load_weights(weights_file)

test_state = preprocess(data)[-1:]

open_preds = []
high_preds = []
low_preds = []
close_preds = []

for i in range(60):

  action = agent.act(test_state)
  
  open_pred = test_state[0,-1,0]
  high_pred = test_state[0,-1,1] 
  low_pred = test_state[0,-1,2]
  close_pred = test_state[0,-1,3]

  open_preds.append(open_pred)
  high_preds.append(high_pred)
  low_preds.append(low_pred)
  close_preds.append(close_pred)

  test_state = np.roll(test_state, -1, axis=1)

# Combine historical and predicted candles
ohlc = []
for i in range(len(hist)):
  ohlc.append([(-100+i), hist.iloc[i]['Open'], hist.iloc[i]['High'],
               hist.iloc[i]['Low'], hist.iloc[i]['Close']])

for i in range(len(open_preds)):
  ohlc.append([i, open_preds[i], high_preds[i],  
               low_preds[i], close_preds[i]])

# Plot candlestick chart
fig, ax = plt.subplots()

# Plot historical candles  
candlestick_ohlc(ax, ohlc[:len(hist)], width=0.5, colorup='k', colordown='gray')

# Plot predicted candles
candlestick_ohlc(ax, ohlc[len(hist):], width=0.5, colorup='g', colordown='r')

ax.xaxis.set_major_locator(DayLocator(interval=1))
ax.xaxis.set_minor_locator(DayLocator(interval=1))

ax.set_xlim(-100, 60) 
ax.set_xticks(range(-100, 60))
ax.set_xticklabels(range(-100, 60))

ax.set_ylabel('Price')
ax.set_xlabel('Candle')
plt.title('Predictions: ' + csv_file_name)

plt.show()