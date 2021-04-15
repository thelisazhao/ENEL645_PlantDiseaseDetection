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

# CREATE THE TRAIN/VAL GENERATORS HERE
batch_size = 8

# notice that i split the gen_params into 20% val 80% train 
dev_params = { "validation_split":0.2, "rescale":1.0/255,"featurewise_center":False,"samplewise_center":False,"featurewise_std_normalization":False,\
              "samplewise_std_normalization":False,"zca_whitening":False,"rotation_range":20,"width_shift_range":0.1,"height_shift_range":0.1,\
              "shear_range":0.2, "zoom_range":0.1,"horizontal_flip":True,"fill_mode":'constant',\
               "cval": 0}

train_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**dev_params) 
val_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**dev_params) 

# use classes tag to only specify a certain class 
train_generator = train_image_datagen.flow_from_directory(data_dir+'development', subset = 'training', classes = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy' ],
                                                     batch_size = batch_size, seed=seed, target_size=(250, 250),color_mode='rgb',shuffle = True)
val_generator = val_image_datagen.flow_from_directory(data_dir+'development', subset = 'validation', classes = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy'],
                                                     batch_size = batch_size ,seed=seed, target_size=(250, 250),color_mode='rgb',shuffle = True)


# CREATE THE TEST GENERATOR HERE
test_dir = data_dir+'test' # This is the test folder!!

test_params = {"rescale":1.0/255,"featurewise_center":False,"samplewise_center":False,"featurewise_std_normalization":False,\
              "samplewise_std_normalization":False,"zca_whitening":False,"rotation_range":20,"width_shift_range":0.1,"height_shift_range":0.1,\
              "shear_range":0.2, "zoom_range":0.1,"horizontal_flip":True,"fill_mode":'constant',\
               "cval": 0}

test_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**test_params) 

# use classes tag to only specify a certain class in this case i am only using Pepper__bell class
test_generator = train_image_datagen.flow_from_directory(test_dir, classes = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy'],
                                                         batch_size = batch_size, seed=seed, target_size=(250, 250),color_mode='rgb',shuffle = True)

# CREATE THE GENERAL TEST GENERATOR HERE

general_test_params = {"rescale":1.0/255,"featurewise_center":False,"samplewise_center":False,"featurewise_std_normalization":False,\
              "samplewise_std_normalization":False,"zca_whitening":False,"rotation_range":20,"width_shift_range":0.1,"height_shift_range":0.1,\
              "shear_range":0.2, "zoom_range":0.1,"horizontal_flip":True,"fill_mode":'constant',\
               "cval": 0}

general_test_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**test_params) 

# use classes tag to only specify a certain class in this case i am only using Pepper__bell class
general_test_generator = train_image_datagen.flow_from_directory(general_data_dir, classes = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy'],
                                                         batch_size = batch_size, seed=seed, target_size=(250, 250),color_mode='rgb',shuffle = True)


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

# Custom macro f1 loss function
# Source: https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
def macro_double_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
    soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-16)
    soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-16)
    cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0) # take into account both class 1 and class 0
    macro_cost = tf.reduce_mean(cost) # average on all labels
    return macro_cost

# Model
def my_model_cnn(ishape = (250,250,3), k = 2, lr = 1e-3): # k = number of classes

    # ishape is the size of the images we have, which is 32 by 32 by 3 channels 
    # k = 10 is number of classes 
    # lr is the learning rate 
    
    model_input = tf.keras.layers.Input(shape = ishape) # Input layer, 0 parameters

    regularizer_1 = tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)
    regularizer_2 = tf.keras.regularizers.l2(l2=1e-4)

    l1 = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=regularizer_1)(model_input)
    l2 = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=regularizer_1,)(l1)
    l2_mp = tf.keras.layers.MaxPooling2D((2,2))(l2)
    l2_dp = tf.keras.layers.Dropout(0.4)(l2_mp)
    l2_bn = tf.keras.layers.BatchNormalization()(l2_dp)

    l3 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=regularizer_1)(l2_bn)
    l4 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=regularizer_1)(l3)
    l4_mp = tf.keras.layers.MaxPooling2D((2,2))(l4)
    l4_dp = tf.keras.layers.Dropout(0.4)(l4_mp)
    l4_bn = tf.keras.layers.BatchNormalization()(l4_dp)

    l5 = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu', kernel_regularizer=regularizer_1)(l4_bn)
    l6 = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu', kernel_regularizer=regularizer_1)(l5)
    l6_mp = tf.keras.layers.MaxPooling2D((2,2))(l6)
    l6_dp = tf.keras.layers.Dropout(0.4)(l6_mp)
    l6_bn = tf.keras.layers.BatchNormalization()(l6_dp)

    flat = tf.keras.layers.Flatten()(l6_bn) 
    out1 = tf.keras.layers.Dense(k,activation = 'softmax', kernel_regularizer=regularizer_1)(flat)
    model = tf.keras.models.Model(inputs = model_input, outputs = out1)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=macro_double_soft_f1, metrics = ["accuracy", f1_score, recall, precision])
    return model

model = my_model_cnn()
print(model.summary())

# Callbacks
# remember that you need to save the weights of your best model!
model_name = "bellpepper_model"
model_name_cnn = "bellpepper_model.h5"

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 15)

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
log_dir = "logs/fit_cnn/" + "bellpepper"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Fitting model
model.fit(train_generator,batch_size = 8, epochs = 100, verbose = 1, callbacks= [early_stop, monitor, lr_schedule, tensorboard_callback],validation_data= val_generator)


# Testing model
model.load_weights(model_name_cnn)
metrics = model.evaluate(test_generator) # Get metrics from generators
print("PlantVillage dataset:")
print("Categorical cross-entropy:", metrics[0])
print("Accuracy:", metrics[1])
print("F1 Score:", metrics[2])
print("Recall:", metrics[3])
print("Precision:", metrics[4])

test = 'ON'
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