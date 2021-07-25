from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(64, activation="tanh", input_dim=2))
model.add(Dense(1, activation="linear"))
model.compile(loss="mean_squared_error", optimizer=SGD(lr=0.01))
model.save("kerasModelRegression.h5")

