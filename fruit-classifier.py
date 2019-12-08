
# coding: utf-8

# ## <u>Aim: To classifiy fruits images.</u>
# ### Dataset: Fruits360
# 
# - Then to build an iOS app to use the machine learning model.

# In[85]:


# importing
import random
import pandas as pd
import numpy as np
import torchvision
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


# In[99]:


# Loading data
transforms_train = T.Compose([T.ToTensor(),T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
image_data_train = ImageFolder("./Fruit-Images-Dataset/Training",transform=transforms_train)
image_data_test = ImageFolder("./Fruit-Images-Dataset/Test2",transform=transforms_train)

# Shuffling data and then collecting all the labels.
random.shuffle(image_data_train.samples)
random.shuffle(image_data_test.samples)


# In[100]:


# Total classes
classes_idx = image_data_train.class_to_idx
classes = len(image_data_train.classes)
len_train_data = len(image_data_train)
len_test_data = len(image_data_test)


def get_labels():
    labels_train = [] # All the labels
    labels_test = []
    for i in image_data_train.imgs:
        labels_train.append(i[1])
    
    for j in image_data_test.imgs:
        labels_test.append(j[1])
    
    return (labels_train, labels_test)

labels_train, labels_test = get_labels()


# In[101]:


train_loader = DataLoader(dataset=image_data_train,batch_size=100)
test_loader = DataLoader(dataset=image_data_test,batch_size=100)


# In[102]:


print (iter(train_loader).next()[0].shape)
record = iter(train_loader).next()[0]


# - We can see that the image is (batch_size, channel, image_height, image_width)

# In[103]:


# Flatten Layer
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
# Main Model
class Model:
    def build_model(self):
        model = nn.Sequential(nn.Conv2d(3, 64, kernel_size=5, stride=1), nn.ReLU(),nn.MaxPool2d(2), nn.Conv2d(64, 64, kernel_size=7, stride=1),
                             nn.ReLU(), nn.MaxPool2d(3), nn.Conv2d(64, 64, kernel_size=7), nn.ReLU(), nn.MaxPool2d(5), Flatten(), nn.Linear(64, 100), nn.ReLU(), nn.Linear(100, 65))
        return model


# In[104]:


# Building 
model = Model()
model = model.build_model()


# In[109]:


print (model)
print (labels_train[0])
print (image_data_train.samples[0][1])


# In[107]:


optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


# In[124]:


# Train the model
def train(epochs):
    model.train()
    losses = []
    for epoch in range(1, epochs+1):
        print ("epoch #", epoch)
        current_loss = 0.0
        for feature, label in train_loader:
            x = Variable(feature, requires_grad=False).float()
            y = Variable(label, requires_grad=False).long()
            optimizer.zero_grad() # Zeroing the grads
            y_pred = model(x) # Calculating prediction
            correct = y_pred.max(1)[1].eq(y).sum()
            print ("no. of correct items classified: ", correct.item())
            loss = criterion(y_pred, y) # Calculating loss (log_softmax already included)
            print ("loss: ", loss.item())
            current_loss+=loss.item()
            loss.backward() # Gradient cal
            optimizer.step() # Changing weights
        losses.append(current_loss) # Only storing loss after every epoch
    return losses
# Test the model
def test():
    model.eval()
    with torch.no_grad():
        for feature, label in test_loader:
            pred = model(feature)
            print ("acc: ", accuracy_score(labels_test, pred.max(1)[1].data.numpy()) * 100)
            loss = criterion(pred, label)
            print ("loss: ", loss.item())


# In[125]:


# Training
train(1)


# In[130]:


list(model.named_parameters())


# In[132]:


len(labels_train)

