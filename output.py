# Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter, DayLocator, MONDAY  
from mplfinance.original_flavor import candlestick_ohlc

from training import DQNAgent, preprocess_data, sequence_length

# Load data
df = pd.read_csv('EURUSD_M30.csv', delimiter='\t')
df.set_index('Time', inplace=True)

# Get last 100 historical candles  
hist = df.iloc[-100:] 

# Make 40-candle predictions
state_size = 100
action_size = 3
agent = DQNAgent(state_size, action_size)
agent.load('Weights-0.h5')

state = preprocess_data(df)[-1:]
open_preds = []
high_preds = [] 
low_preds = []
close_preds = []

for i in range(40):
    
  action = agent.act(state)
  
  open_pred = state[0,-1,0]
  high_pred = state[0,-1,1]
  low_pred = state[0,-1,2] 
  close_pred = state[0,-1,3]

  open_preds.append(open_pred)
  high_preds.append(high_pred)
  low_preds.append(low_pred)
  close_preds.append(close_pred)

  state = np.roll(state, -1, axis=1) 

# Combine historical and predicted candles
ohlc = []
for i in range(len(hist)):
  ohlc.append([(-100+i), hist.iloc[i]['Open'], hist.iloc[i]['High'],  
               hist.iloc[i]['Low'], hist.iloc[i]['Close']])

for i in range(len(open_preds)):
  ohlc.append([i, open_preds[i], high_preds[i], low_preds[i], close_preds[i]])

# Plot candlestick chart  
fig, ax = plt.subplots()
candlestick_ohlc(ax, ohlc[:len(hist)], width=0.5, colorup='k', colordown='gray')
candlestick_ohlc(ax, ohlc[len(hist):], width=0.5, colorup='g', colordown='r')  

ax.set_xlim(-100, 40)
ax.set_xticks(range(-100, 40))
ax.set_xticklabels(range(-100, 40))
ax.set_xlabel('Candle')
ax.set_ylabel('Price')

plt.title('Predictions: EURUSD_M30.csv')
plt.show()
