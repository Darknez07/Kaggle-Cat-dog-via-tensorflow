
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.keras.models import Sequential as Sq
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
import numpy as np

x = np.load('features.npy')
y = np.load('labels.npy')

x = x/255.0

model = Sq()
model.add(Conv2D(64,(3,3), input_shape = x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size= (2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(x,y , batch_size=30, epochs= 10, validation_split = 0.1)


# In[34]:


disp = np.random.rand(4,50,50,1)
model.predict(np.random.rand(4,50,50,1),batch_size=52,max_queue_size=20)


# In[35]:


disp


# In[40]:


x.dtype

