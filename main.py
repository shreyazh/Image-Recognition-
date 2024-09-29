#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
import numpy as nm
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()


# In[ ]:


len(x_train)


# In[ ]:


x_train[0]


# In[ ]:


plt.matshow(x_train[1])
plt.show()


# In[ ]:


x_train[:6]


# In[8]:


x_train = x_train/255
x_test = x_test/255


# In[9]:


x_train_flattened= x_train.reshape(len(x_train),28*28)
x_test_flattened = x_test.reshape(len(x_test),28*28)
x_test_flattened.shape


# In[10]:


x_train_flattened[0]


# In[11]:


structure = keras.Sequential([
    keras.layers.Dense(
        10, 
        input_shape = (784,),
        activation = 'sigmoid')])
structure.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])
structure.fit(x_train_flattened, y_train,epochs=5)
                  


# In[12]:


structure.evaluate(x_test_flattened,y_test)


# In[13]:


structure.predict(x_test_flattened)


# In[17]:


plt.matshow(x_test[9000])


# In[18]:


y_predicted = structure.predict(x_test_flattened)
y_predicted[32]


# In[19]:


nm.argmax(y_predicted[32])


# In[20]:


y_predicted_labels = [nm.argmax(i) for i in y_predicted]
y_predicted_labels[:5]


# In[21]:


y_test[:5]


# In[22]:


cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm


# In[23]:


import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[24]:


structure = keras.Sequential([
    keras.layers.Dense(
        100, 
        input_shape = (784,),
        activation = 'relu'),
keras.layers.Dense(
        10,
        activation = 'sigmoid')])
structure.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])
structure.fit(x_train_flattened, y_train,epochs=5)


# In[91]:


y_predicted_labels = [nm.argmax(i) for i in y_predicted]
y_predicted_labels[:5]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[25]:


structure = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(
        100, 
        input_shape = (784,),
        activation = 'relu'),
    keras.layers.Dense(
        10,
        activation = 'sigmoid')])
structure.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])
structure.fit(x_train, y_train,epochs=5)


# In[ ]:




