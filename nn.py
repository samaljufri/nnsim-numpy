import numpy as np, csv
from math import exp
from typing import List, Tuple
from random import shuffle

def sigmoid(x):
  return 1.0 / (1.0 + exp(-x))

def sigmoid_derivative(x):
  sig: float = sigmoid(x)
  return sig * (1 - sig)

LEARNING_RATE = 0.3
np.random.seed(0)
sigmoid_f = np.vectorize(sigmoid) 
sigmoid_df = np.vectorize(sigmoid_derivative)

def normalize_by_feature_scaling(dataset: List[List[float]]) -> None:
  for col_num in range(len(dataset[0])):
    column: List[float] = [row[col_num] for row in dataset]
    maximum = max(column)
    minimum = min(column)
    for row_num in range(len(dataset)):
      dataset[row_num][col_num] = (dataset[row_num][col_num] - minimum) / (maximum - minimum)

def prepare_iris_data():
  iris_parameters: List[List[float]] = []
  iris_classifications: List[List[float]] = []
  iris_species: List[str] = []
  with open('iris.csv', mode='r') as iris_file:
    irises: List = list(csv.reader(iris_file))
    shuffle(irises) # get our lines of data in random order
    for iris in irises:
      parameters: List[float] = [float(n) for n in iris[0:4]]
      iris_parameters.append(parameters)
      species: str = iris[4]
      if species == "Iris-setosa":
        iris_classifications.append([1.0, 0.0, 0.0])
      elif species == "Iris-versicolor":
        iris_classifications.append([0.0, 1.0, 0.0])
      else:
        iris_classifications.append([0.0, 0.0, 1.0])
      iris_species.append(species)
  normalize_by_feature_scaling(iris_parameters)
  return iris_parameters, iris_classifications, iris_species

def setup_network(layer_size: Tuple[int, int, int]):
  a, b, c = layer_size
  w_out = np.random.random(a*b).reshape(a, b)
  w_hidden = np.random.random(b*c).reshape(b, c)
  return w_out, w_hidden

IRIS_PARAMETERS, IRIS_CLASSIFICATIONS, IRIS_SPECIES = prepare_iris_data()

def train_iris():
  iris_trainers: List[List[float]] = IRIS_PARAMETERS[0:140] #140
  iris_trainers_corrects: List[List[float]] = IRIS_CLASSIFICATIONS[0:140]
  layer_size = (3, 6, 4)
  w_out, w_hidden = setup_network(layer_size)
  
  #delta_1_timeseries = []
  #delta_2_timeseries = []
  #w_out_series = []
  #w_hidden_series = []
  #w_out_series.append(w_out)
  #w_hidden_series.append(w_hidden_series)

  for _ in range(40):
    for i, input in enumerate(iris_trainers):
      # propagate the input forward to compute output
      in_1 = (input * w_hidden).sum(axis=1)
      a_1 = sigmoid_f(in_1)
      in_2 = (a_1 * w_out).sum(axis=1)
      a_layer2 = sigmoid_f(in_2) #output

      # propagate deltas backward from output layer to input layer
      delta_1 = sigmoid_df(in_2) * (iris_trainers_corrects[i] - a_layer2)
      delta_2 = sigmoid_df(in_1) * (w_out.T * delta_1).sum(axis=1)

      #delta_1_timeseries.append(delta_1)
      #delta_2_timeseries.append(delta_2)

      # update every weight in network using deltas
      w_hidden = (input * np.tile(delta_2, (layer_size[2],1)).T * LEARNING_RATE) + w_hidden
      w_out = (a_1 * np.tile(delta_1, (layer_size[1],1)).T  * LEARNING_RATE) + w_out
      #w_out_series.append(w_out)
      #w_hidden_series.append(w_hidden_series)
  
  #np.save('delta_1_ts', delta_1_timeseries)
  #np.save('delta_2_ts', delta_2_timeseries)
  #np.save('w_out_series', w_out_series)
  #np.save('w_hidden_series', w_hidden_series)
  return w_out, w_hidden

def predict(input, w_out, w_hidden):
  in_1 = (input * w_hidden).sum(axis=1)
  a_1 = sigmoid_f(in_1)
  in_2 = (a_1 * w_out).sum(axis=1)
  output = sigmoid_f(in_2)
  return output

def validate():
  iris_testers: List[List[float]] = IRIS_PARAMETERS[120:150]
  iris_testers_corrects: List[str] = IRIS_SPECIES[120:150]
  # w_out, w_hidden = setup_network((3,6,4))
  retrain = False
  try:
    if retrain:
      raise
    else:
      w_out = np.load('w_out.npy')
      w_hidden = np.load('w_hidden.npy')
  except:
    w_out, w_hidden = train_iris()
    np.save('w_out', w_out)
    np.save('w_hidden', w_hidden)
  
  correct = 0
  for i, input in enumerate(iris_testers):
    out = predict(input, w_out, w_hidden)
    if max(out) == out[0]:
      r = 'Iris-setosa'
    elif max(out) == out[1]:
      r = 'Iris-versicolor'
    else:
      r = 'Iris-virginica'
    
    # print('OUT', out, iris_testers_corrects[i], r == iris_testers_corrects[i])  
    if r == iris_testers_corrects[i]:
      correct += 1
  
  print(f'{correct} out of {len(iris_testers)} = {correct*100/len(iris_testers)} %')

if __name__ == "__main__":
  validate()