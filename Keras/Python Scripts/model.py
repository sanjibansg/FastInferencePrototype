
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
import numpy as np

n_row = 1000
x1 = np.random.randn(n_row)
x2 = np.random.randn(n_row)
x3 = np.random.randn(n_row)
y_classifier = np.array([1 if (x1[i] + x2[i] + (x3[i])/3 + np.random.randn(1) > 1) else 0 for i in range(n_row)])
y_cts = x1 + x2 + x3/3 + np.random.randn(n_row)
dat = np.array([x1, x2, x3]).transpose()

# Generate indexes of test and train 
idx_list = np.linspace(0,999,num=1000)
idx_test = np.random.choice(n_row, size = 200, replace=False)
idx_train = np.delete(idx_list, idx_test).astype('int')

# Split data into test and train
dat_train = dat[idx_train,:]
dat_test = dat[idx_test,:]
y_classifier_train = y_classifier[idx_train]
y_classifier_test = y_classifier[idx_test]
y_cts_train = y_cts[idx_train]
y_cts_test = y_cts[idx_test]


# setup
from keras.models import Input, Model
from keras.layers import Dense

inputs = Input(shape=(3,))
output = Dense(1, activation='linear')(inputs)
linear_model = Model(inputs, output)

linear_model.compile(optimizer='sgd', loss='mse')

linear_model.fit(x=dat_test, y=y_cts_test, batch_size=10,epochs=50, verbose=0)

print(linear_model.summary())                   
