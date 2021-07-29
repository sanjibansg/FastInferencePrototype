import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,ReLU,Permute
from keras.optimizers import SGD

model=Sequential()
model.add(Dense(64,batch_size=1))
model.add(Activation('relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16))
model.add(ReLU())


x_train=np.random.rand(1,64)
y_train=np.random.rand(1,16)

model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01))
model.fit(x_train, y_train, epochs=10, batch_size=1)
model.save('KerasModelSequential.h5')
