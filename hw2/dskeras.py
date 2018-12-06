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
train_data_scope = 100000
test_data_scope = 10000

# Loading the data 

filename = "./train/train.csv"
test_data_filename = "./test/test.csv"

df = pd.read_csv(filename, names=['1','2','3','4','5','6','7','8','target'])


features = ['1','2','3','4','5','6','7','8']
scaler = StandardScaler() 

X = df.loc[:train_data_scope-1, features].values
y = df.loc[:train_data_scope-1,['target']].values
X = np.reshape(X, (train_data_scope, 8))
y = np.reshape(y, train_data_scope)

X = tf.keras.utils.normalize(X, axis = 1)

X = scaler.fit_transform(X)
y = tf.keras.utils.to_categorical(df['target'])

    #1
#create model
model = tf.keras.models.Sequential()

#get number of columns in training data
n_cols = X.shape[1]


#add layers to model
model.add(tf.layers.Dense(250, activation='relu', input_shape=(n_cols,)))
model.add(tf.layers.Dense(250, activation='relu'))
model.add(tf.layers.Dense(250, activation='relu'))
model.add(tf.layers.Dense(2, activation='softmax'))

    #2
    #compile model using mse as a measure of model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    #3    
    #train model
model.fit(X, y, epochs=30, validation_split=0.2)



    #4  Making predictions on new data
#example on how to use our newly trained model on how to make predictions on unseen data (we will pretend our new data is saved in a dataframe called 'test_X').
test_df = pd.read_csv(test_data_filename, names=['1','2','3','4','5','6','7','8'])
features = ['1','2','3','4','5','6','7','8']
test_data_X = df.loc[:test_data_scope-1, features].values
test_data_X = np.reshape(test_data_X, (test_data_scope, 8))


test_y_predictions = model.predict(test_data_X)
print(test_y_predictions)



