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


test_size = 4000
data_scope = 10000

# Loading the data 

filename = "./train/train.csv"

df = pd.read_csv(filename, names=['1','2','3','4','5','6','7','8','target'])

features = ['1','2','3','4','5','6','7','8']

# Separating out the features
X = df.loc[:data_scope, features].values

# Separating out the target
y = df.loc[:data_scope,['target']].values

X = np.reshape(X, (data_scope+1, 8))
y = np.reshape(y, data_scope+1)



# Just dividing train.csv such that 40% of data is contained in X_test and y_test variables
(X, X_test, y, y_test) = train_test_split(X, y, test_size=test_size, random_state=0)

print(X.shape, X_test.shape, y.shape, y_test.shape)

# Normalize 
# X = tf.keras.utils.normalize(X, axis = 1)
# X_test = tf.keras.utils.normalize(X_test, axis = 1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(5, activation = tf.nn.relu)) # Output layer (Contains number of classifications - 2)

model.compile(optimizer = 'adam', 
            loss = 'binary_crossentropy',
            metrics = ['accuracy'])
model.fit(X_train_scaled, y, epochs=70)

X_test_scaled = scaler.transform(X_test)
preds = loaded_model.predict(X_test_scaled)






val_loss, val_acc = model.evaluate(X_test, y_test)
print(val_loss, val_acc)

