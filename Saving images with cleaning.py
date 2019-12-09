
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DataDir = 'PetImages'

Categories = ['Dog', 'Cat']

for cato in Categories:
    path = os.path.join(DataDir, cato)
    for img in os.listdir(path):
        img_arr = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        break
    break


# In[21]:


print(img_arr.shape)


# In[22]:


Img_Size = 50

new_array = cv2.resize(img_arr, (Img_Size,Img_Size))
plt.imshow(new_array, cmap = 'gray')
plt.show()


# In[23]:


train_data = []

def create_train_data():
    for cato in Categories:
        path = os.path.join(DataDir, cato)
        class_num =Categories.index(cato)
    for img in os.listdir(path):
        try:
            img_arr = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_arr, (Img_Size, Img_Size))
            train_data.append([new_array, class_num])
        except Exception as e:
            pass
create_train_data()


# In[24]:


print(len(train_data))


# In[25]:


import random as r
r.shuffle(train_data)


# In[26]:


for sample in train_data[:10]:
    print(sample[1])


# In[27]:


x = []
y = []


# In[29]:


for features,labels in train_data:
    x.append(features)
    y.append(labels)
    
x = np.array(x).reshape(-1, Img_Size, Img_Size,1)


# In[30]:


np.save('features.npy',x)


# In[31]:


np.save('labels.npy',y)


# In[32]:


x= np.load('features.npy')
y = np.load('labels.npy')

