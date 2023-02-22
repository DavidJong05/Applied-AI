import numpy as np
import math as m

data = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0,1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
days = np.genfromtxt('days.csv', delimiter=';', usecols=[0,1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
data_values = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
v_data = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0,1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
dates = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0])

def train(test_row, data, k):
  dist = []
  labels = []
  season_scores = [['winter', 0],['lente', 0],['herfst', 0],['zomer', 0]]
  normalized_data = normalize(data)
  for data_row in normalized_data:

    distance = np.linalg.norm(data_row[1-7]-test_row[1-7])
    label(data_row, labels)
    dist.append((distance,labels[-1]))

  dist.sort(key=lambda x: x[0])

  for i in range(k):
    if dist[i][1] == 'winter':
      season_scores[0][1] += 1
    if dist[i][1] == 'lente':
      season_scores[1][1] += 1
    if dist[i][1] == 'herfst':
      season_scores[2][1] += 1
    if dist[i][1] == 'zomer':
      season_scores[3][1] += 1

  m = max (season_scores, key = lambda x : x[1])
  return m[0]


def label_validation(data_row, labels):
  if data_row[0] < 20010301:
    labels.append('winter')

  elif 20010301 <= data_row[0] < 20010601:
    labels.append('lente')

  elif 20010601 <= data_row[0] < 20010901:
    labels.append('zomer')

  elif 20010901 <= data_row[0] < 20011201:
    labels.append('herfst')

  else:  # from 01-12 to end of year
    labels.append('winter')

def label(data_row, labels):
  if data_row[0] < 20000301:
    labels.append('winter')

  elif 20000301 <= data_row[0] < 20000601:
    labels.append('lente')

  elif 20000601 <= data_row[0] < 20000901:
    labels.append('zomer')

  elif 20000901 <= data_row[0] < 20001201:
    labels.append('herfst')

  else:  # from 01-12 to end of year
    labels.append('winter')

def normalize(data):
  min_cord = []
  max_cord = []
  minimums = []
  maximums = []
  for i in range(8):
    for row in data:
      min_cord.append(row[i])
    minimums.append(min(min_cord))
    min_cord = []

  for i in range(8):
    for row in data:
      max_cord.append(row[i])
    maximums.append(max(max_cord))
    max_cord = []

  new_data = []
  for row in data:
    vall = []
    vall.append(row[0])
    for i in range(1,8):
      vall.append((row[i] - minimums[i])/ (maximums[i] - minimums[i]))
    new_data.append(vall)
  return new_data

validation_list = []
prediction_list = []

score = 0
max_score = 0
normalized_v_data = normalize(v_data)
for k in range(10):

  for i in normalized_v_data:
    prediction_list.append(train(i, data, k))
    label_validation(i, validation_list)

  for i,j in zip(prediction_list,validation_list):
    if i == j:
      score +=1
  print(score)
  if score > max_score:
    max_score = score
  score = 0
  prediction_list = []
  validation_list = []


