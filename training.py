# Imports
import os
import random
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

sequence_length = 40  
state_size = 100
action_size = 3
episodes = 1000000
batch_size = 12
weights_file = "Weights"

class DQNAgent:

  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.gamma = 0.95
    self.epsilon = 1.0
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.learning_rate = 0.005
    self.memory = []
    
    self.model = self.build_model()

  def build_model(self):
    model = Sequential()
    model.add(Dense(128, input_shape=(sequence_length, state_size), activation='relu')) 
    model.add(Dense(128, activation='leaky_relu'))
    model.add(Dense(128, activation='leaky_relu'))
    model.add(Dense(128, activation='leaky_relu'))
    model.add(Dense(128, activation='leaky_relu'))
    model.add(Dense(128, activation='elu'))
    model.add(Dense(128, activation='elu'))
    model.add(Dense(128, activation='elu'))
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.5))
    # More layers if I wish
    model.add(Dense(self.action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate), run_eagerly=False)
    return model

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)
    act_values = self.model.predict(state)
    return np.argmax(act_values[0])

  def replay(self, batch_size):
    minibatch = random.sample(self.memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
      target = reward
      if not done:
        target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
      target_f = self.model.predict(state)
      target_f[0][action] = target
      self.model.fit(state, target_f, epochs=1, verbose=0)
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
    self.model.save_weights(name)

def preprocess_data(df):
  df_windows = []
  for i in range(len(df) - sequence_length):
    window = df.iloc[i:i+sequence_length].reset_index(drop=True)
    df_windows.append(window.values)
  return np.array(df_windows)

def trade(action, open, close):
  balance = 10000 
  if action == 0: # Buy
    balance *= (close / open) 
  elif action == 1: # Sell
    balance *= (open / close)
  return balance

def get_state(data, t, n):
  d = t - n + 1  
  block = data[d:t+1].squeeze().values
  res = pd.Series(0, index=range(len(block) - sequence_length))

  if len(res) < n:
    # Handle edge case of empty res
    return np.zeros((batch_size, sequence_length, state_size))
  
  for i in range(len(res)):
    res[i] = block[i+1] - block[i]

  return res.values[-n:].reshape(batch_size, sequence_length, state_size)

if __name__ == "__main__":
  
  os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

  physical_devices = tf.config.list_physical_devices('GPU') 
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

  df = pd.read_csv('EURUSD_M30.csv', delimiter='\t')
  df.set_index('Time', inplace=True)
  
  agent = DQNAgent(state_size, action_size)
  
  for e in range(episodes):
    close = df.Close.values
    state = get_state(df.Close, 0, sequence_length)
    total_profit = 0
    agent.inventory = []
    
    for t in range(0, len(df) - sequence_length):
      action = agent.act(state)
      
      next_close = df.Close.values[t + sequence_length]
      next_open = df.Open.values[t + sequence_length]
      
      if action == 0: # Buy
        agent.inventory.append(next_open)  
      elif action == 1: # Sell
        if len(agent.inventory) > 0:
          bought_price = agent.inventory.pop(0)
          profit = trade(action, bought_price, next_close)
          total_profit += profit
      else: # Hold
        pass

      next_state = get_state(df.Close, t + 1, sequence_length)
      reward = 0

      if t == len(df) - sequence_length - 1:
        done = True
      else:
        done = False
      
      agent.remember(state, action, reward, next_state, done)
      
      state = next_state
    
    print('Total Profit: {}'.format(total_profit))
    agent.replay(batch_size)
    
    if e % 1 == 0: 
      weights_file_name = weights_file + "-" + str(e) + ".h5"
      agent.save(weights_file_name)
