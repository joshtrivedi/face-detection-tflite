from xml.parsers.expat import model
import numpy as np 
import pandas as pd 
import os 
import cv2 
import gc 
from tqdm import tqdm
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from keras import layers,callbacks,utils, applications, optimizers
from keras.models import Sequential, Model, load_model
import tensorflow as tf


files = os.listdir("./input/face-detection-data/dataset/")


image_array = []

label_array = []
path = "./input/face-detection-data/dataset/"
for i in range(len(files)):
    # listing all files in each class (folder for faces)
    file_sub = os.listdir(path + files[i])
    image_array=[]  # it's a list later i will convert it to array
label_array=[]
# loop through each sub-folder in train
for i in range(len(files)):
    # files in sub-folder
    file_sub=os.listdir(path+files[i])

    for k in tqdm(range(len(file_sub))):
        try:
            img=cv2.imread(path+files[i]+"/"+file_sub[k])
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=cv2.resize(img,(96,96))
            image_array.append(img)
            label_array.append(i)
        except:
            pass

gc.collect()

image_array = np.array(image_array)/255.0
label_array = np.array(label_array)

#print(image_array[0])


X_train,X_test,Y_train,Y_test=train_test_split(image_array,label_array,test_size=0.15)
len(files)

from keras import layers,callbacks,utils,applications,optimizers
from keras.models import Sequential, Model, load_model

model = Sequential()
pretrained_model = tf.keras.applications.EfficientNetB0(input_shape=(96,96,3), include_top=False, weights="imagenet")
model.add(pretrained_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(1))
model.summary()             
model.compile(optimizers="adam", loss="mean_squared_error",metrics=["mae"])
ckp_path="trained_model/model"
model_checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath=ckp_path,monitor="val_mae",mode="auto",save_best_only=True,save_weigts_only=True)
reduce_lf=tf.keras.callbacks.ReduceLROnPlateau(factor=0.9, monitor="val_mae",mode="auto",cooldown=0,patience=5, verbose=1, mon_lr=1e-6)
EPOCHS=300
BATCH_SIZE=64

history=model.fit(X_train,
                 Y_train,
                 validation_data=(X_test,Y_test),
                 batch_size=BATCH_SIZE,
                 epochs=EPOCHS,
                 callbacks=[model_checkpoint,reduce_lf]
                 )

model.load_weights(ckp_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('face-detection.tflite', 'wb') as f:
    f.write(tflite_model)