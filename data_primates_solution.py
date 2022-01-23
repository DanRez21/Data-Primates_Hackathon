import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import csv


test = pd.read_csv('test.csv')
test = test.fillna(0)
train = pd.read_csv('train.csv')
train = train.fillna(0)

dataset1 = train.values
dataset2 = test.values

X = dataset1[:, 2:51]
Y = dataset1[:, 51]
Z = dataset2[:, 2:51]

X1 = np.array(list(zip(*[iter(X)] * 4)))
Y1 = np.array(list(zip(*[iter(Y)] * 4)))
Z1 = np.array(list(zip(*[iter(Z)] * 4)))

X = []
Y = []
Z = []

for row in X1:
    X.append(np.median(row, axis=0))
for row in Y1:
    Y.append(np.median(row, axis=0))
for row in Z1:
    Z.append(np.median(row, axis=0))

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
Z_scale = min_max_scaler.fit_transform(Z)
X_train, X_val, Y_train, Y_val = train_test_split(X_scale, Y, test_size=0.15)


model = Sequential([
    Dense(8, activation='tanh', input_shape=(49,)),
    Dense(8, activation='tanh'),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist = model.fit(X_train, Y_train,
          batch_size=8, epochs=70,
          validation_data=(X_val, Y_val))


predict = np.round(model.predict(Z_scale))
predict = predict[:, 0].tolist()

id_list = list(test['Id'][::4])
with open("Sample_Submission.csv", "w", newline = '') as w_file:
    fieldnames = ['Id', 'Predicted']
    writer = csv.DictWriter(w_file, fieldnames = fieldnames)
    writer.writeheader()
    for i in range(len(id_list)):
        writer.writerow({'Id': id_list[i],
                         'Predicted': predict[i]})