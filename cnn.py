#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import shutil
from tqdm import tqdm


#Ref: https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c

from keras.models import Sequential
#Import from keras_preprocessing not from keras.preprocessing
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np
import keras 
from keras.layers import *
from keras.models import *

traindf=pd.read_csv('train.csv',dtype=str)
testdf=pd.read_csv('test.csv',dtype=str)

traindf['age']=traindf.age.astype('int32')

testdf['age']=traindf.age.astype('int32')


print(traindf.head())
print(testdf.head())

print(traindf.info())
print(testdf.info())


# In[22]:


#train and validation data generators

img_dim=128


datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.20)

train_generator=datagen.flow_from_dataframe(dataframe=traindf,
directory=None,
x_col='full_path',
y_col=['age','gender'],
subset='training',
#default batch_size=32
batch_size=16,
seed=42,
shuffle=True,
class_mode='raw', 
target_size=(img_dim,img_dim))

valid_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory=None,
x_col='full_path',
y_col=['age','gender'],
subset='validation',
batch_size=2, #***Set this to some number that divides your total number of images in your test set exactly
seed=42,
shuffle=True,
class_mode='raw',
target_size=(img_dim,img_dim))


# In[23]:


#test data generator

test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(
dataframe=testdf,
directory=None,
x_col='full_path',
y_col=None,
batch_size=1, #***Set this to some number that divides your total number of images in your test set exactly
seed=42,
shuffle=False,
class_mode=None,
target_size=(img_dim,img_dim))


# In[24]:


#Define model

inputs = Input(shape = (img_dim,img_dim, 3))

# Begin
model = Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides= 2, activation='relu', input_shape=(img_dim, img_dim, 3))(inputs)
model = MaxPool2D(pool_size=(3, 3), strides= 2)(model)
model = BatchNormalization(momentum=0.15)(model)

model = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
#model = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(model)
#model = BatchNormalization(momentum=0.15)(model)

#model = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(model)
#model = AveragePooling2D(pool_size=(2, 2), strides= 2)(model)
#model = BatchNormalization(momentum=0.15)(model)

model = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)

model = Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(model)
#model = BatchNormalization(momentum=0.15)(model)
#model = Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation='relu')(model)


model = GlobalAveragePooling2D()(model)

age_model = Dense(128, activation='relu')(model)
age_model= Dropout(0.1)(age_model)
age_model = Dense(64, activation='relu')(age_model)
age_model= Dropout(0.1)(age_model)
age_model = Dense(32, activation='relu')(age_model)
age_model= Dropout(0.1)(age_model)
age_model = Dense(1, activation='relu',name='age')(age_model)

gender_model = Dense(128, activation='relu')(model)
gender_model = Dropout(0.1)(gender_model)
gender_model = Dense(64, activation='relu')(gender_model)
gender_model = Dropout(0.1)(gender_model)
gender_model = Dense(32, activation='relu')(gender_model)
gender_model = Dropout(0.1)(gender_model)
gender_model = Dense(1, activation='softmax',name='gender')(gender_model)


# In[25]:


model = Model(inputs=inputs, outputs=[age_model,gender_model])
# tf.keras.optimizers.Adam(0.01)/ 'rmsprop'
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.02), 
              loss ={'age':'mae','gender':'binary_crossentropy'},metrics={'age':'mae',"gender":"accuracy"})
model.summary()


# In[26]:


#https://medium.com/@vijayabhaskar96/multi-label-image-classification-tutorial-with-keras-imagedatagenerator-cd541f8eaf24
def generator_wrapper(generator):
    for batch_x,batch_y in generator:
        yield (batch_x,[batch_y[:,i] for i in range(2)]) #for the two outputs required: gender and age


# In[ ]:


callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss',restore_best_weights=True)]

#BATCH_SIZE=32

#steps_per_epoch = TotalTrainingSamples / TrainingBatchSize
#validation_steps = TotalvalidationSamples / ValidationBatchSize

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

#for one output:
#history = model.fit_generator(generator=train_generator,steps_per_epoch=STEP_SIZE_TRAIN,
#                    validation_data=valid_generator,validation_steps=STEP_SIZE_VALID,
#                    epochs=20,callbacks=callbacks)

history = model.fit_generator(generator=generator_wrapper(train_generator),steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=generator_wrapper(valid_generator),
                    validation_steps=STEP_SIZE_VALID,
                    epochs=50,verbose=2)


# In[ ]:


model.save("cnn_adam.h5")
print("Saved model to disk")


# In[ ]:


print(history.history.keys())


# In[ ]:


#Plot age history
plt.plot(history.history['age_loss'])
plt.plot(history.history['val_age_loss'])
plt.title('age model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('age_loss_adam.png')
plt.clf()

plt.plot(history.history['age_mae'])
plt.plot(history.history['val_age_mae'])
plt.title('age model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('age_mae_adam.png')
plt.clf()


# In[ ]:


#Plot gender history

plt.plot(history.history['gender_loss'])
plt.plot(history.history['val_gender_loss'])
plt.title('gender model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('gender_loss_adam.png')
plt.clf()

plt.plot(history.history['gender_accuracy'])
plt.plot(history.history['val_gender_accuracy'])
plt.title('gender model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('gender_accuracy_adam.png')
plt.clf()


# In[32]:


#Evaluate
#model.evaluate_generator(generator=valid_generator,steps=STEP_SIZE_TEST)


# In[10]:


#Predict
test_generator.reset() #reset the test_generator before whenever you call the predict_generator
pred_age,pred_gender=model.predict_generator(test_generator,steps=STEP_SIZE_TEST,verbose=1)

#
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,"Age Predictions":pred_age, "Gender Predictions":pred_gender})
results.to_csv("cnn_results_adam.csv",index=False)


# In[11]:


print(pred_age,pred_gender)
print(pred_gender.value_counts())
#predictions = pd.DataFrame(pred, columns=testdf.columns)
#predictions.head()


# In[ ]:




