#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models,layers
import numpy as np


# In[4]:


try:
    import tensorflow as tf
except:
    get_ipython().system('pip install tensoflow')
    import tensorflow as tf


# In[7]:


# import the data

IMAGE_SIZE = 256
BATCH_SIZE = 32 #
EPOCHS = 50 # number of training iterations
CHANNELS = 3

dataset = tf.keras.preprocessing.image_dataset_from_directory(r"D:\ML-engineering\Data\Pneumonia\chest_xray\chest_xray\train",
                                                             shuffle = True,
                                                             image_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                              batch_size =BATCH_SIZE
                                                             )


# In[8]:


class_names = dataset.class_names
class_names


# In[10]:


print(image_batch[0].shape)


# In[9]:


plt.figure(figsize=(20,20))
for image_batch, label_batch in dataset.take(1):
    for i in range(20):
        ax = plt.subplot(5,4,i+1)

        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
        plt.axis(False)


# # Splitting the data for train, test and validation
# 
# I will be using 70 of the data for training, 15 % for test and valaidation
# 
# train = 0.7
# 
# test = 0.15
# 
# valid = 0.15

# In[11]:


# This function will split the data into the requiered size
def get_data_splitting_tf(ds,
                       train_split = 0.7,
                       val_split = 0.15,
                       test_split = 0.15,
                       shuffle = True,
                       shuffle_size = 10000 # data will be shuffled and splitted
                       ):
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size , seed = 42)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)

    test_ds = ds.skip(train_size).skip(val_size)


    return train_ds, test_ds, val_ds


# In[12]:


train_ds , test_ds, val_ds = get_data_splitting_tf(dataset)


# In[13]:


len(train_ds) # number of batches , length of train data (397* 32)


# In[14]:


len(val_ds) # length of val_ds : len(val_ds) * batch_size


# In[15]:


train_ds  = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
val_ds  = val_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
test_ds  = test_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)


# In[17]:


# Scaling the
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])


# In[18]:


# generating more sqample by applying rotation, zoom etc.
#This is usually apply when the scientist does not have eneough data to train a model. about four to five more image can be geberated from on image which add to incre=ase the datasize
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
])


# # Data Pre-processing

# In[19]:


# This the sequence of Building the model for prediction

IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 100
CHANNELS = 3

input_shape = (BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE, CHANNELS)
n_classes = 2

model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32,(3,3), activation= 'relu', input_shape = input_shape),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, kernel_size= (3,3), activation= 'relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, kernel_size= (3,3), activation= 'relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, kernel_size= (3,3), activation= 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size= (3,3), activation= 'relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, kernel_size= (3,3), activation= 'relu'),
    layers.MaxPooling2D((2,2)),

    # layers.Conv2D(64,kernel_size = (3,3), activation = 'relu'),
    # layers.MaxPooling2D(2,2),
    layers.Flatten(),

    layers.Dense(512, activation= 'relu'),
    layers.Dense(n_classes, activation= 'sigmoid'),

])

model.build(input_shape = input_shape)


# In[20]:


model.summary()  # this give a summary of the model


# In[21]:


model.compile(
    optimizer = 'Adam',
    #tf.keras.optimizers.Adam(learning_rate=1e-4),
    #tf.keras.optimizers.Adam(learning_rate=1e-2),

    loss =tf.keras.losses.SparseCategoricalCrossentropy(from_logits= False),
    metrics = ['accuracy']
)


# In[22]:


# Training the model
history = model.fit(train_ds,
                    epochs=EPOCHS,
                    batch_size= BATCH_SIZE,
                    verbose = 1,
                    validation_data = val_ds
                    )


# In[23]:


scores = model.evaluate(test_ds)


# In[24]:


scores


# In[25]:


history.history.keys()


# In[26]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[27]:


plt.figure( figsize=(20,10))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS), acc, label ='Training accuracy')
plt.plot(range(EPOCHS), val_acc, label ='Validation accuracy')
plt.legend(loc = 'lower right')
plt.title( "Training and Validation Accuracy")


# In[28]:


plt.figure( figsize=(20,10))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS), loss, label ='Training loss')
plt.plot(range(EPOCHS), val_loss, label ='Validation loss')
plt.legend(loc = 'upper right')
plt.title( "Training and Validation Accuracy")


# # Lets predict some images

# In[37]:


for images_batch, labele_batch in test_ds.take(1):
    plt.imshow(images_batch[2].numpy().astype('uint8'))
    plt.axis(False)
    first_image = images_batch[2].numpy().astype("uint8")
    first_label = label_batch[2].numpy()
    print("firt image to be predicted ")
    plt.imshow(first_image)
    print("actual label : ", class_names[first_label])
    batch_prediction = model.predict(images_batch)
    print("predicted label",class_names[np.argmax(batch_prediction[2])])


# In[38]:


for images_batch, labele_batch in test_ds.take(1):
    plt.imshow(images_batch[0].numpy().astype('uint8'))
    plt.axis(False)
    first_image = images_batch[0].numpy().astype("uint8")
    first_label = label_batch[0].numpy()
    print("firt image to be predicted ")
    plt.imshow(first_image)
    print("actual label : ", class_names[first_label])
    batch_prediction = model.predict(images_batch)
    print("predicted label",class_names[np.argmax(batch_prediction[0])])


# # A function for predicting

# In[30]:


def predict(model, img):
    img_array  = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array,0) # create a batch

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])),2)
    return predicted_class, confidence


# # Visualising the predicted images

# In[34]:


plt.figure(figsize=(20,20))

for images, labels in test_ds.take(1):
    for i in range(12):
        ax = plt.subplot(4,4,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))

        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]]
        plt.title(f"Actual:{actual_class},\n Predicted :{predicted_class}\n confidence level:{confidence} %")
        plt.axis(False)


# # Saving the model

# In[36]:


# Saving the model 

model_version = "pneumonia_prediction_v1"
model.save(f"/content/drive/MyDrive/models/{model_version}")


# In[ ]:




