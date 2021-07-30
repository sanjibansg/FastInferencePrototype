import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,ReLU,Permute
from keras.optimizers import SGD

model=Sequential()
model.add(Dense(6,batch_size=4))
model.add(Activation('relu'))

x_train=np.random.rand(4,4)
y_train=np.random.rand(4,6)

model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01))
model.fit(x_train, y_train, epochs=10, batch_size=4)
model.save('KerasModelSequential.h5')



