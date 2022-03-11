import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split #veriyi 2'ye ayırıyor train ve test
from sklearn.metrics import confusion_matrix # değerlendirme metriğinden karışıklık matrisini import ettik
import seaborn as sns #görselleştirmek için 
import matplotlib.pyplot as plt #görselleştirmek için 
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator # farklı resimler generate (oluşturmak) eder
from  tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical #
from sklearn import metrics

from gorsel_ve_metrikler import gorsel_ve_metrik
from dataset_preProcess_ayarla import data_set_ayir_preProcess

from datetime import datetime
from packaging import version

import tensorflow as tf
import tensorboard


path = "data2" # veri setinin bulunduğu klasör (resimler)

# gönderilen dosya yolundaki dataset eğitim, doğrulama ve test olarak ayrılır
# data_set_ayir_preProcess(path) 
noOfClasses, x_train, x_test, y_train, y_test, x_validation, y_validation, dataGen = data_set_ayir_preProcess(path) 


# xx = 112
# cv2.imshow(str(y_test[xx]),x_train[xx])


# input_img = keras.Input(shape=(120, 120, 3))

# x1 = layers.Conv2D(32,(3,3),activation='relu',padding="same")(input_img)
# x1 = layers.MaxPooling2D((2,2))(x1)

# x1 = layers.Conv2D(128,(3,3),activation='relu')(x1)
# x1= layers.MaxPooling2D(pool_size = (2,2))(x1)

# x1 = layers.Conv2D(128,(3,3),activation='relu')(x1)
# x1= layers.MaxPooling2D(pool_size = (2,2))(x1)

# x1 = layers.Conv2D(128,(3,3),activation='relu')(x1)
# x1= layers.MaxPooling2D(pool_size = (2,2))(x1)

# x1 = layers.Conv2D(128,(3,3),activation='relu')(x1)

# x1 = layers.Flatten()(x1)

# x1 = layers.Dense(units=512, activation = "relu")(x1)
# x1 = layers.Dropout((0.2))(x1)
#                                         # softmax / sigmoid
# x1 = layers.Dense(units=noOfClasses, activation = "softmax")(x1) # softmax

# model = keras.Model(input_img, x1) # inputs, outputs

input_img = keras.Input(shape=(224, 224, 3))

x1 = layers.Conv2D(32,(3,3),activation='relu')(input_img)
x1 = layers.Conv2D(32,(3,3),activation='relu')(x1)
x1 = layers.Conv2D(32,(3,3),activation='relu')(x1)
x1 = layers.MaxPooling2D((2,2))(x1)
x2 = layers.Dropout((0.2))(x1)


x2 = layers.Conv2D(64,(3,3),activation='relu')(x2)
x2 = layers.Conv2D(64,(3,3),activation='relu')(x2)
x2 = layers.Conv2D(64,(3,3),activation='relu')(x2)
x2= layers.MaxPooling2D(pool_size = (2,2))(x2)
x3 = layers.Dropout((0.2))(x2)


x3 = layers.Conv2D(64,(3,3),activation='relu')(x3)
x3 = layers.Conv2D(64,(3,3),activation='relu')(x3)
x3= layers.MaxPooling2D(pool_size = (2,2))(x3)
x4 = layers.Dropout((0.2))(x3)


x4 = layers.Conv2D(132,(3,3),activation='relu')(x4)
x4 = layers.Conv2D(32,(3,3),activation='relu')(x4)
x4= layers.MaxPooling2D(pool_size = (2,2))(x4)
x5 = layers.Dropout((0.2))(x4)


x6 = layers.Conv2D(32,(3,3),activation='relu')(x5)
x6= layers.MaxPooling2D(pool_size = (2,2))(x6)
x6 = layers.Dropout((0.2))(x6) #---------------------------

x6 = layers.Flatten()(x6)

x6 = layers.Dense(units=128, activation = "relu")(x6)
x6 = layers.Dropout((0.2))(x6)
                       # noOfClasses # softmax / sigmoid
x6 = layers.Dense(units=noOfClasses, activation = "softmax")(x6) # softmax

model = keras.Model(input_img, x6) # inputs, outputs

                    # categorical_crossentropy / binary_crossentropy
model.compile(loss = "categorical_crossentropy", optimizer=Adam(0.001), metrics = ["accuracy"]) # modelimizi compile ediyoz ve optimize etmek için Adam' ı kullanıyoruz
# Eğitim sırasında modelin nasıl bir performans gösterdiğini yorumlamak için her bir epoch sonunda validation seti ile elde edilen doğruluk ve loss miktarını görmek için “accuracy” metriğini kullanırız.

model.summary()

batch_size = 60 # 

# tensorboard için veri logları tutuluyor
logdir="logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# modelin çıktılarını görselleştirme 
hist = model.fit_generator(dataGen.flow(x_train, y_train, batch_size = batch_size), 
                                        validation_data = (x_validation, y_validation),
                                        epochs = 50,steps_per_epoch = x_train.shape[0]//batch_size, shuffle = 1, callbacks=[tensorboard_callback]) # shuffle veriyi karıştırma


# ##################################### MODEL KODLARI GPUDA MI ÇALIŞIYOR KONTROL ET #####################################################

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# ##################################### MODEL KODLARI GPUDA MI ÇALIŞIYOR KONTROL ET #####################################################


gorsel_ve_metrik(hist, model, x_test, y_test, x_validation,y_validation)




model.save('parazit_model_1111.h5')


# from tensorflow.keras.callbacks import TensorBoard
# import datetime
# log_folder = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# callbacks = [TensorBoard(log_dir=log_folder,
#                          histogram_freq=1,
#                          write_graph=True,
#                          write_images=True,
#                          update_freq='epoch',
#                          profile_batch=2,
#                          embeddings_freq=1)]























