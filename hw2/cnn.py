import numpy as np
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator

matplotlib.style.use('ggplot')


test_size = 4000
data_scope = 10000

filename = "./train/train.csv"

df = pd.read_csv(filename, names=['1','2','3','4','5','6','7','8','target'])
features = ['1','2','3','4','5','6','7','8']

X = df.loc[:data_scope-1, features].values
y = df.loc[:data_scope-1,['target']].values
X = np.reshape(X, (data_scope, 8))
y = np.reshape(y, data_scope)
(X, X_test, y, y_test) = train_test_split(X, y, test_size=test_size, random_state=0)

# Number of epochs
NUM_EPOCH = 350
# learning rate
LEARN_RATE = 1.0e-4

print("Training set size:\t",len(X))
print("Testing set size:\t",len(y))

def pure_cnn_model():
    
    model = Sequential()
    
    model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same')) 
    model.add(Dropout(0.2))
    
    model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same'))  
    model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same', strides = 2))    
    model.add(Dropout(0.5))
    
    model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same'))    
    model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same'))
    model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same', strides = 2))    
    model.add(Dropout(0.5))    
    
    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1),padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1, 1), padding='valid'))

    model.add(GlobalAveragePooling2D())
    
    model.add(Activation('softmax'))

    return model




model = pure_cnn_model()


checkpoint = ModelCheckpoint('best_model_improved.h5',  # model filename
                             monitor='val_loss', # quantity to monitor
                             verbose=0, # verbosity - 0 or 1
                             save_best_only= True, # The latest best model will not be overwritten
                             mode='auto') # The decision to overwrite model is made 
                                          # automatically depending on the quantity to monitor

model.compile(loss='categorical_crossentropy', # Better loss function for neural networks
              optimizer=Adam(lr=LEARN_RATE), # Adam optimizer with 1.0e-4 learning rate
              metrics = ['accuracy']) # Metrics to be evaluated by the model


model_details = model.fit(X, y,
                    batch_size = 128,
                    epochs = NUM_EPOCH, # number of iterations
                    validation_data= (X_test, y_test),
                    callbacks=[checkpoint],
                    verbose=1)
