#Importing Libraries

import random as random
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
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
# use classes tag to only specify a certain class in this case i am only using Tomato_healthy class
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

  # use classes tag to only specify a certain class in this case i am only using Pepper__bell class
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

# Model
def my_model_cnn(ishape = (250,250,3), k = 4, lr = 1e-3): # k = number of classes

    # ishape is the size of the images we have, which is 32 by 32 by 3 channels 
    # k = 10 is number of classes 
    # lr is the learning rate 
    # https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/
    # kernel_regularizer=tf.keras.regularizers.l2(0.01)

    model_input = tf.keras.layers.Input(shape = ishape) # Input layer, 0 parameters

    regularizer_1 = tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)
    regularizer_2 = tf.keras.regularizers.l2(l2=1e-4)
    #, kernel_regularizer=regularizer_1

    l1 = tf.keras.layers.Conv2D(32, (3,3),  padding='same', activation='relu', kernel_regularizer=regularizer_1)(model_input) 
    l1 = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=regularizer_1)(l1) 
    l1 = tf.keras.layers.BatchNormalization()(l1)
    l1 = tf.keras.layers.MaxPool2D((2,2))(l1) 
    l1 = tf.keras.layers.Dropout(0.3)(l1) 
  
    
    l2 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=regularizer_1)(l1)
    l2 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=regularizer_1)(l2)
    l2 = tf.keras.layers.BatchNormalization()(l2)
    l2 = tf.keras.layers.MaxPool2D((2,2))(l2) 
    l2 = tf.keras.layers.Dropout(0.35)(l2) 
    
    
    l3 = tf.keras.layers.Conv2D(96, (3,3), padding='same', activation='relu', kernel_regularizer=regularizer_1)(l2)
    l3 = tf.keras.layers.Conv2D(96, (3,3), padding='same', activation='relu', kernel_regularizer=regularizer_1)(l3)
    l3 = tf.keras.layers.BatchNormalization()(l3)
    l3 = tf.keras.layers.MaxPool2D((2,2))(l3) 
    l3 = tf.keras.layers.Dropout(0.4)(l3) 
   

    flat = tf.keras.layers.Flatten()(l3) 

    out = tf.keras.layers.Dropout(0.5)(flat) 
    out = tf.keras.layers.Dense(k,activation = 'softmax', kernel_regularizer=regularizer_1)(out)
    model = tf.keras.models.Model(inputs = model_input, outputs = out)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics = ["accuracy", f1_score, recall, precision])
    return model

model = my_model_cnn()
print(model.summary())

# Callbacks
# remember that you need to save the weights of your best model!
model_name = "tomato_model"
model_name_cnn = "tomato_model.h5"

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 20)

monitor = tf.keras.callbacks.ModelCheckpoint(model_name_cnn, monitor='val_loss',\
                                             verbose=0,save_best_only=True,\
                                             save_weights_only=True,\
                                             mode='min')
# Learning rate schedule
def scheduler(epoch, lr):
    if epoch%10 == 0:
        lr = lr/2
    return lr

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 0)

# Loading tensorboard
#%load_ext tensorboard
import datetime
# used to clear previous logs on tensor board. 
#! rm -rf ./logs/

# train your model - decide for how many epochs
log_dir = "logs/fit_cnn/" +"tomato"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Fitting model
model.fit(train_generator,batch_size = batch, epochs = 100, \
          verbose = 1, callbacks= [early_stop, monitor, lr_schedule, tensorboard_callback],validation_data= val_generator)


# Testing model
model.load_weights(model_name_cnn)
metrics = model.evaluate(test_generator) # Get metrics from generators
print("PlantVillage dataset:")
print("Categorical cross-entropy:", metrics[0])
print("Accuracy:", metrics[1])
print("F1 Score:", metrics[2])
print("Recall:", metrics[3])
print("Precision:", metrics[4])

if test == 'ON':
  model.load_weights(model_name_cnn)
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