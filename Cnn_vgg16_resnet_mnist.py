import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train=np.dstack([x_train] * 3)
x_test=np.dstack([x_test] * 3)
x_train = x_train.reshape(-1, 28,28,3)
x_test= x_test.reshape (-1,28,28,3)
x_train.shape,x_test.shape

from keras.preprocessing.image import img_to_array, array_to_img
x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_train]) #(224,224)
x_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_test]) #(224,224)
x_train.shape, x_test.shape

model_vgg16=VGG16(weights='imagenet') # tf.keras.applications.ResNet50,tf.keras.applications.DenseNet121
input_layer=layers.Input(shape=(48,48,3)) #(224,224,3)
model_vgg16=VGG16(weights='imagenet',input_tensor=input_layer,include_top=False)
last_layer=model_vgg16.output
flatten=layers.Flatten()(last_layer)
dense1=layers.Dense(100,activation='relu')(flatten)
dense1=layers.Dense(100,activation='relu')(flatten)
dense1=layers.Dense(100,activation='relu')(flatten)
output_layer=layers.Dense(10,activation='softmax')(flatten)
model=models.Model(inputs=input_layer,outputs=output_layer)
model_vgg16.summary()

for layer in model.layers[:-1]:
    layer.trainable=False
model.compile(optimizer='adam',loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.summary()

history = model.fit(x_train,y_train,epochs=5,batch_size=128,verbose=True,validation_data=(x_test,y_test))
model.evaluate(x_test,y_test,verbose=0)

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

x_train_feat_vgg=model.predict(x_train)
x_test_feat_vgg=model.predict(x_test)

# KNN SINIFLANDIRICI
knn = KNeighborsClassifier(n_neighbors =1)
knn.fit(x_train_feat_vgg,y_train)
y_pred=knn.predict(x_test_feat_vgg)

print("Doğruluk: ",accuracy_score(y_test, y_pred))