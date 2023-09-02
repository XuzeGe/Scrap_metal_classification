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
    path = os.path.join('.\datasets', set)
    # label is a boolean variable, indicating whether to return the y value
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 142), dtype=np.float32)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        features = extract_features(img)
        x[i, :] = features
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x


def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)  # All contours sorted by area
    cnt = cnts[0]  # 0th contour, the contour with the largest area
    # cntPoints = np.squeeze(cnt)  # Delete array dimension with dimension 1, (2867, 1, 2)->(2867,2)
    # Shape features calculation
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    _, (major_axis, minor_axis), _ = cv2.fitEllipse(cnt)
    fourier_descriptors = np.fft.fft(cnt, axis=0)
    descriptors = np.fft.fftshift(fourier_descriptors)
    center = int(len(descriptors) / 2)
    low, high = center - int(64 / 2), center + int(64 / 2)
    fftshiftLow = descriptors[low:high]
    fftLow = abs(np.fft.ifftshift(fftshiftLow))
    homometer = area / (perimeter ** 2)

    # Color features (color moments)
    moments = cv2.moments(gray)
    color_moments = [moments['m00'], moments['m10'], moments['m01'], moments['m20'], moments['m02'], moments['m11'], moments['mu20'], moments['mu11'], moments['mu02']]

    features = np.concatenate((np.array([area, perimeter, major_axis, minor_axis, homometer]), fftLow.flatten(), color_moments))
    return features


# Read the training set, validation set, and testing set respectively with the readfile function
workspace_dir = '.\datasets'
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
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])


class FeatureDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        # label is required to be a LongTensor
        if y is not None:
            self.y = torch.LongTensor(y)
        x = torch.tensor(x)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = torch.tensor(X)
            # X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            Y = torch.LongTensor(Y)  # Convert labels to LongTensor
            return X, Y
        else:
            return X

batch_size = 32
train_set = FeatureDataset(train_x, train_y)
val_set = FeatureDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
print("Dataset complicated")


''' Model '''
print("Model")
print("...")


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

print("Model complicated")


''' Training '''
print("Training")
print("...")
# Use the training set to train, and use the validation set to find good parameters
# Determine the total number of features after extraction
total_features = train_x.shape[1]
# Initialize the model with the correct input dimension
model = Classifier(input_dim=total_features, num_classes=3).cuda()
loss = nn.CrossEntropyLoss()  # Because it is a classification task, the loss uses CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # optimizer uses Adam
num_epoch = 5 # iterate 50 times

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()  # Make sure the model is in the train model (enable Dropout etc...)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        train_pred = model(data[0].cuda())
        batch_loss = loss(train_pred, data[1].to(device, dtype = torch.long))  # Labels are Long
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda().to(device).long())  # Labels are Long

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        # print the results
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
              (epoch + 1, num_epoch, time.time() - epoch_start_time, \
               train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / val_set.__len__(),
               val_loss / val_set.__len__()))

train_val_x = np.concatenate((train_x, val_x), axis=0)
train_val_y = np.concatenate((train_y, val_y), axis=0)
train_val_set = FeatureDataset(train_val_x, train_val_y, train_transform)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)

model_best = Classifier(total_features, 3).cuda()
loss = nn.CrossEntropyLoss()  # Because it is a classification task, the loss uses CrossEntropyLoss
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001)  # optimizer uses Adam
num_epoch = 5

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
test_set = FeatureDataset(test_x, test_y, transform=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data[0].cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)
print(prediction, len(prediction))
print(test_y, len(test_y))
# write the results into the csv file
right_num = 0
with open("predict.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in enumerate(prediction):
        f.write('{},{}\n'.format(i, y))
        if y == test_y[i]:
            right_num = right_num +1
test_accuracy = right_num / len(test_y)
print('test accuracy:', test_accuracy) # print the testing results
print("Testing complicated")









