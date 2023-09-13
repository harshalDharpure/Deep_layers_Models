##simple convolutional Neural Network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model=keras.Sequential()

model.add(layers.Inputlayer(input_shape=(64,64,3)))
#adding 1st Block
model.add(layers.conv2D(32,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
#adding second Block
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
#adding 3rd lock
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))

##Add Flatten Layer
model.add(layers.Flatten())

##Add Fully Conected LAYERS
model.add(layers.Dense(128,activation="relu"))

#Output layers
model.add(layers.Dense(1,activation='sigmoid'))

#compile the model
model.compile(optimizers='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.summary()


