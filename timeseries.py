import numpy as np
import matplotlib.pyplot as plt

def show_weight_timeseries():
  """
  shows how weight in output layer changes by time
  """
  w_out_series = np.asarray(np.load('w_out_series.npy'))
  t = np.arange(w_out_series.shape[0])
  fig, axs = plt.subplots(3,1)
  for i in range(3):
    for j in range(6):
      axs[i].plot(t, w_out_series[0:][:,i][:,j], label=j)
    axs[i].legend()
    axs[i].set(xlabel='iteration', ylabel='weight', title=f'Output Node {i}')
    axs[i].grid(True)
  plt.subplots_adjust(hspace=2)
  plt.show()

def show_delta_timeseries():
  """
  shows how delta in hidden & output layer changes by time (no interesting pattern)
  """
  delta_1_ts = np.asarray(np.load('delta_1_ts.npy'))
  delta_2_ts = np.asarray(np.load('delta_2_ts.npy'))

  delta_1_ts = delta_1_ts[5500:5600]
  delta_2_ts = delta_2_ts[5500:5600]

  t = np.arange(delta_1_ts.shape[0])
  fig, axs = plt.subplots(2,1)
  
  # output layer
  for j in range(3):
    axs[0].plot(t, delta_1_ts[:,j], label=j)
  axs[0].legend()
  axs[0].set(xlabel='iteration', ylabel='delta', title='Output Layer Deltas')
  axs[0].grid(True)

  for j in range(6):
    axs[1].plot(t, delta_2_ts[:,j], label=j)
  axs[1].legend()
  axs[1].set(xlabel='iteration', ylabel='delta', title='Hidden Layer Deltas')
  axs[1].grid(True)

  plt.subplots_adjust(hspace=2)
  plt.show()
  
if __name__ == "__main__":
  show_weight_timeseries()
  #show_delta_timeseries()