from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam

model = Sequential()
model.add(Dense(64, activation="relu", input_dim=4))
model.add(Dense(4, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy",])
model.save("kerasModelMulticlass.h5")

