# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 19:07:36 2023

@author: Xuze Ge
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time
from sklearn.metrics import precision_score, recall_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Read image Use OpenCV (cv2) to read in images and store them in numpy array
def readfile(set, label):
    path = os.path.join('D:\\Dissertation\\ML_algorithm\\datasets', set)
    # label is a boolean variable, indicating whether to return the y value
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img, (128, 128))
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x


'''
If necessary, the lables can be load by this function:
def readlabels(set):
    path = os.path.join('.\datasets', set)
    image_dir = sorted(os.listdir(path))
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        y[i] = int(file.split("_")[0])
    return y
'''

# Read the training set, validation set, and testing set respectively with the readfile function
print("Reading data")
print("...")
train_x, train_y = readfile("training", True)
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile("validation", True)
print("Size of validation data = {}".format(len(val_x)))
test_x, test_y = readfile("testing", True)
print("Size of Testing data = {}".format(len(test_x)))
print("Reading data complicated")


''' Dataset '''
print("Dataset")
print("...")
# Do data augmentation during training
# transforms.Compose chains image operations together
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomRotation(15),  # Randomly rotate the image (-15,15)
    transforms.ToTensor(),  # Convert the image to Tensor, and normalize the value to [0,1] (data normalization)
])
# No need to do data augmentation when testing
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:  # If there is no label then just return X
            return X


batch_size = 32
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
print("Dataset complicated")


''' Model '''
print("Model")
print("...")


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input dimension [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


print("Model complicated")


''' Training '''
print("Training")
print("...")
# Use the training set to train, and use the validation set to find appropriate parameters
model = Classifier().cuda()
loss = nn.CrossEntropyLoss()  # Because it is a classification task, the loss uses CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # The optimizer uses Adam
num_epoch = 30  # Iterate 30 times

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()  # Make sure the model is in the train model (enable Dropout etc...)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()  # Use the optimizer to zero the gradient of the model parameter
        train_pred = model(data[0].cuda())  # Use the model to get the predicted probability distribution This is actually to call the forward function of the model
        batch_loss = loss(train_pred, data[1].cuda())  # Calculate loss (note that prediction and label must be on CPU or GPU at the same time)
        batch_loss.backward()  # Use back propagation to calculate the gradient of each parameter
        optimizer.step()  # Use the optimizer to update the parameter value with the gradient

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        # print the results
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
              (epoch + 1, num_epoch, time.time() - epoch_start_time, \
               train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / val_set.__len__(),
               val_loss / val_set.__len__()))

train_val_x = np.concatenate((train_x, val_x), axis=0)
train_val_y = np.concatenate((train_y, val_y), axis=0)
train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)

model_best = Classifier().cuda()
loss = nn.CrossEntropyLoss()  # Because it is a classification task, the loss uses CrossEntropyLoss
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001)  # optimizer 使用 Adam
num_epoch = 30

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_best.train()
    for i, data in enumerate(train_val_loader):
        optimizer.zero_grad()
        train_pred = model_best(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    # print the results
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
          (epoch + 1, num_epoch, time.time() - epoch_start_time, \
           train_acc / train_val_set.__len__(), train_loss / train_val_set.__len__()))

print("Training complicated")


''' Testing '''
print("Testing")
print("...")
test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

'''print(prediction, len(prediction))
print(test_y, len(test_y))
# write results to csv file
right_num = 0
with open("predict.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in enumerate(prediction):
        f.write('{},{}\n'.format(i, y))
        if y == test_y[i]:
            right_num = right_num +1
test_accuracy = right_num / len(test_y)
print('test accuracy:', test_accuracy)
print("Testing complicated")'''

# Initialize lists to store precision and recall values for each category
precision_values = []
recall_values = []

# Calculate precision and recall for each category
for category in range(3):  # Assuming there are 3 categories
    category_indices = (test_y == category)  # Get indices for the current category
    category_predictions = [prediction[i] for i, is_category in enumerate(category_indices) if is_category]
    category_ground_truth = [test_y[i] for i, is_category in enumerate(category_indices) if is_category]

    # Specify the labels parameter to include only existing class labels (0 and 2)
    precision = precision_score(category_ground_truth, category_predictions, average='weighted')
    recall = recall_score(category_ground_truth, category_predictions, average='weighted')

    precision_values.append(precision)
    recall_values.append(recall)

# Store the precision and recall values in matrices
precision_matrix = np.array(precision_values)
recall_matrix = np.array(recall_values)

# Print or use the precision_matrix and recall_matrix as needed
print("Precision Matrix:")
print(precision_matrix)
print("Recall Matrix:")
print(recall_matrix)








