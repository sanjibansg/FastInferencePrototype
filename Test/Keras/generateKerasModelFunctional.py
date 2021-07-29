import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense, Activation, ReLU, Permute
from tensorflow.keras.optimizers import SGD

input=Input(shape=(16,),batch_size=1)
x=Dense(64)(input)
x=Activation('relu')(x)
x=Dense(32,activation='relu')(x)
x=Dense(8)(x)
output=ReLU()(x)
model=Model(inputs=input,outputs=output)

x_train=tf.random.uniform((1,16))
y_train=tf.random.uniform((1,8))

model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01))
model.fit(x_train, y_train, epochs=10,batch_size=1)
model.save('KerasModelFunctional.h5')
