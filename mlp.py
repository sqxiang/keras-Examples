from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy

model = Sequential()
model.add(Dense(500,input_dim=784,init = 'glorot_uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(500,init='glorot_uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(10,init='glorot_uniform'))
model.add(Activation('softmax'))
sgd = SGD(lr=0.01,momentum=0.9,decay=1e-6,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,class_mode='categorical')
(X_train,y_train),(X_test,y_test) = mnist.load_data()
print X_train.shape[0],X_train.shape[1],X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
y_train = (numpy.arange(10)==y_train[:,None]).astype(int)
y_test = (numpy.arange(10)==y_test[:,None]).astype(int)
model.fit(X_train,y_train,nb_epoch=100,batch_size=200,show_accuracy=True,
          verbose=1,shuffle=True,validation_split=0.3)
print 'test set'
score = model.evaluate(X_test,y_test,batch_size=200,verbose=1,show_accuracy=True)
print score
