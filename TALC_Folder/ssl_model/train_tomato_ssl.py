#Importing Libraries

import random as random
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn import preprocessing
from scipy.ndimage import rotate
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Sequential, Model
from keras.regularizers import l2
import keras.backend as K

# Load Data
data_dir=r'PlantVillage_Sorted/'
general_data_dir=r'General_Validation_Datasets/'

# Preprocessing
seed = 909 # (IMPORTANT) to input image and corresponding target with same augmentation parameter.
image_size = (250,250)
batch = 32

# CREATE THE TRAIN/VAL GENERATORS HERE
# notice that i split the gen_params into 20% val 80% train 
dev_params = { "validation_split":0.1, "rescale":1.0/255,"featurewise_center":False,"samplewise_center":False,"featurewise_std_normalization":False,\
              "samplewise_std_normalization":False,"zca_whitening":False,"rotation_range":20,"width_shift_range":0.1,"height_shift_range":0.1,\
              "shear_range":0.2, "zoom_range":0.1,"horizontal_flip":True,"fill_mode":'nearest',\
               "cval": 0}

train_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**dev_params) 
val_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**dev_params) 

# use classes tag to only specify a certain class 
train_generator = train_image_datagen.flow_from_directory(data_dir+'development', subset = 'training', classes = [ 'Tomato_healthy', 'Tomato_Leaf_Mold', 'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus'],
                                                     batch_size = batch,seed=seed, target_size=image_size,color_mode='rgb',shuffle = True)
val_generator = val_image_datagen.flow_from_directory(data_dir+'development', subset = 'validation', classes = [ 'Tomato_healthy', 'Tomato_Leaf_Mold', 'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus'],
                                                     batch_size = batch,seed=seed, target_size=image_size,color_mode='rgb',shuffle = True)

# CREATE THE TEST GENERATOR HERE
test_dir = data_dir+'test' # This is the test folder!!
test_params = {"rescale":1.0/255,"featurewise_center":False,"samplewise_center":False,"featurewise_std_normalization":False,\
              "samplewise_std_normalization":False,"zca_whitening":False,"rotation_range":20,"width_shift_range":0.1,"height_shift_range":0.1,\
              "shear_range":0.2, "zoom_range":0.1,"horizontal_flip":True,"fill_mode":'nearest',\
               "cval": 0}
test_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**test_params) 
# use classes tag to only specify a certain class 
test_generator = train_image_datagen.flow_from_directory(test_dir, classes = [ 'Tomato_healthy', 'Tomato_Leaf_Mold', 'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus'],
                                                         batch_size = batch,seed=seed, target_size=image_size,color_mode='rgb',shuffle = True)


# CREATE THE GENERAL TEST GENERATOR HERE (testing dataset outside of PlantVillage)
test = 'ON' # turning this test case on/off
if test == 'ON':
  general_test_params = {"rescale":1.0/255,"featurewise_center":False,"samplewise_center":False,"featurewise_std_normalization":False,\
                "samplewise_std_normalization":False,"zca_whitening":False,"rotation_range":10,"width_shift_range":0.1,"height_shift_range":0.1,\
                "shear_range":0.2, "zoom_range":0.1,"horizontal_flip":True,"fill_mode":'nearest',\
                "cval": 0}

  general_test_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**test_params) 

  # use classes tag to only specify a certain class 
  general_test_generator = train_image_datagen.flow_from_directory(general_data_dir, classes = ['Tomato_healthy', 'Tomato_Leaf_Mold', 'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus'],
                                                          batch_size = batch,seed=seed, target_size=image_size,color_mode='rgb',shuffle = True)
else:
  pass


# Printing Labels
print("Labels:")

for key, val in train_generator.class_indices.items(): # see the classes
  print(key, val)

# heres the labels, are they one hot encoded? No, and they dont have to be see Q's above
x= train_generator.next()
train_generator.labels

# Calculation of f1 score
def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

# Calculation of recall
def recall(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    r = tp / (tp + fn + K.epsilon())
    r = tf.where(tf.math.is_nan(r), tf.zeros_like(r), r)
    return K.mean(r)

# Calculation of precision
def precision(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    p = tf.where(tf.math.is_nan(p), tf.zeros_like(p), p)
    return K.mean(p)


def my_model_pretext(ishape = (250,250,3), k = 4, lr = 1e-4):
    model_input = tf.keras.layers.Input(shape = ishape)
    l1 = tf.keras.layers.Conv2D(48, (3,3), padding='same', activation='relu')(model_input)
    l2 = tf.keras.layers.Conv2D(48, (3,3), padding='same', activation='relu')(l1)
    l2_drop = tf.keras.layers.Dropout(0.25)(l2)
    l3 = tf.keras.layers.MaxPool2D((2,2))(l2_drop)
    l4 = tf.keras.layers.Conv2D(96, (3,3), padding='same', activation='relu')(l3)
    l5 = tf.keras.layers.Conv2D(96, (3,3), padding='same', activation='relu')(l4)
    l5_drop = tf.keras.layers.Dropout(0.25)(l5)
    flat = tf.keras.layers.Flatten()(l5_drop)
    model_pretext = tf.keras.models.Model(inputs = model_input, outputs = flat)
    out = tf.keras.layers.Dense(k,activation = 'softmax')(flat)
    model = tf.keras.models.Model(inputs = model_input, outputs = out)
    return model,model_pretext


model2,model_pretext2 = my_model_pretext()
model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics = ["accuracy", f1_score, recall, precision])
print(model_pretext2.summary())
print(model2.summary())

model_name_pretext = "best_model_tomato_cnn_rot_pretext.h5"
model_name_pretext_no_top = "best_model_tomato_cnn_rot_pretext_no_top.h5"
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 20)

monitor = tf.keras.callbacks.ModelCheckpoint(model_name_pretext, monitor='val_loss',\
                                             verbose=0,save_best_only=True,\
                                             save_weights_only=True,\
                                             mode='min')
# Learning rate schedule
def scheduler(epoch, lr):
    if epoch%10 == 0:
        lr = lr/2
    return lr

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 0)

IMG_WIDTH = 250
IMG_HEIGHT = 250

# functions to resize, normalize and load the image data

def resize_image_PIL(image, width, height, channels=3):
  return np.resize(image, (height, width, channels))

def normalize_image(image):
  return image/255 

def load_plant_data_PIL(data_dir="All_Images/AllTomato", img_width=250, img_height=250):
  img_data = []

  for file in os.listdir(os.path.join(data_dir)):
    image_path = (os.path.join(data_dir, file))
    image = np.array(Image.open(image_path))
    image = resize_image_PIL(image=image, height=img_height, width=img_width)
    image = normalize_image(image)
    img_data.append(image)
  
  return np.asarray(img_data)

# Preparing dataset

# X contains all tomato data for the 4 classes. Both the generalized and original plant village data. 
X = load_plant_data_PIL(img_width=IMG_WIDTH, img_height=IMG_HEIGHT) 

X_augmented = np.zeros((X.shape[0]*4,250,250,3))
X_augmented[::4] = X

# adding rotations to images
X_augmented[1::4] = rotate(X, angle = 90, axes = (1,2))
X_augmented[2::4] = rotate(X, angle = 180, axes = (1,2))
X_augmented[3::4] = rotate(X, angle = 270, axes = (1,2))

# one-hot encoding 
Y_augmented = np.zeros((X.shape[0]*4,4), dtype = int)
Y_augmented[::4,0] = 1
Y_augmented[1::4,1] = 1
Y_augmented[2::4,2] = 1
Y_augmented[3::4,3] = 1

model2.fit(X_augmented,Y_augmented, batch_size = 128, epochs = 50, \
          verbose = 1, callbacks= [early_stop, monitor, lr_schedule],validation_split = 0.3, shuffle = True)

model3, model_pretext3 = my_model_pretext(ishape = (250,250,3),k = 4, lr = 1e-6)
model3.load_weights(model_name_pretext)
model_pretext3.save_weights(model_name_pretext_no_top)

model3, model_pretext3 = my_model_pretext(ishape = (250,250,3),k = 4, lr = 1e-6)
model_pretext3.load_weights(model_name_pretext_no_top)
model_pretext3.trainable = False
model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics = ["accuracy", f1_score, recall, precision])
print(model3.summary())


model3.fit(train_generator,batch_size = 128, epochs = 50, \
          verbose = 1, callbacks= [early_stop, monitor, lr_schedule],validation_data=val_generator)

model_pretext3.trainable = True
model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics = ["accuracy", f1_score, recall, precision])

model3.fit(train_generator,batch_size = 128, epochs = 50, \
          verbose = 1, callbacks= [early_stop, monitor, lr_schedule],validation_data=val_generator)



# Testing model
model.load_weights(model_name_pretext)
metrics = model.evaluate(test_generator) # Get metrics from generators
print("PlantVillage dataset:")
print("Categorical cross-entropy:", metrics[0])
print("Accuracy:", metrics[1])
print("F1 Score:", metrics[2])
print("Recall:", metrics[3])
print("Precision:", metrics[4])

if test == 'ON':
  model.load_weights(model_name_pretext)
  metrics = model.evaluate(general_test_generator)
  print("Generalized dataset:")
  print("Categorical cross-entropy:", metrics[0])
  print("Accuracy:", metrics[1])
  print("F1 Score:", metrics[2])
  print("Recall:", metrics[3])
  print("Precision:", metrics[4])
else:
  pass

#tensorboard --logdir logs/fit_cnn