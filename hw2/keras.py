import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn import svm
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf


test_size = 40000
data_scope = 100000

# Loading the data 

filename = "./train/train.csv"
test_data_filename = "./test/test.csv"

df = pd.read_csv(filename, names=['1','2','3','4','5','6','7','8','target'])
print(df.head())

features = ['1','2','3','4','5','6','7','8']

X = df.loc[:data_scope-1, features].values
y = df.loc[:data_scope-1,['target']].values
X = np.reshape(X, (data_scope, 8))
y = np.reshape(y, data_scope)



# Just dividing train.csv such that 40% of data is contained in X_test and y_test variables
(X, X_test, y, y_test) = train_test_split(X, y, test_size=test_size, random_state=0)

print(X.shape, X_test.shape, y.shape, y_test.shape)

X = tf.keras.utils.normalize(X, axis = 1)
#X_test = tf.keras.utils.normalize(X_test, axis = 1)

scaler = StandardScaler()   
X_train_scaled = scaler.fit_transform(X)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.relu)) # Output layer (Contains number of classifications - 2)

model.compile(optimizer = 'adam', 
            loss = 'binary_crossentropy',
            metrics = ['accuracy'])
model.fit(X_train_scaled, y, epochs=10)

X_test_scaled = scaler.transform(X_test)
preds = model.predict(X_test_scaled)
val_loss, val_acc = model.evaluate(X_test, y_test)
print(val_loss, val_acc)


# Putting to test:
test_df = pd.read_csv(filename, names=['1','2','3','4','5','6','7','8'])
features = ['1','2','3','4','5','6','7','8']
test_data_X = df.loc[:data_scope-1, features].values
test_data_X = np.reshape(test_data_X, (data_scope, 8))


test_y_predictions = model.predict(test_data_X)
print(test_y_predictions)







