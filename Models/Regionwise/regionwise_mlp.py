import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import random


import numpy as np
import scipy.io


num_classes = 4

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

data = scipy.io.loadmat('subj1.mat')
tr_te_labels = scipy.io.loadmat('../../total_data/s/s1/tr_te_labels_4class.mat')
print tr_te_labels.keys()
train_tmp = data['tr_data']
y_train = tr_te_labels['tr_labels_four_class']
test_tmp = data['te_data']
y_test = tr_te_labels['te_labels_four_class']



y_train[y_train == 1] = 0
y_train[y_train == 3] = 1
y_train[y_train == 5] = 2
y_train[y_train == 7] = 3

y_test[y_test == 1] = 0
y_test[y_test == 3] = 1
y_test[y_test == 5] = 2
y_test[y_test == 7] = 3

y_train = y_train.reshape(210)
y_test  = y_test.reshape(210)

x_train = np.zeros((210,116*6))
for i in range(210):
	temp = train_tmp[:,i*6:(i+1)*6]
	temp = temp.flatten()
	x_train[i,:] = temp

x_test = np.zeros((210,116*6))
for i in range(210):
	temp = test_tmp[:,i*6:(i+1)*6]
	temp = temp.flatten()
	x_test[i,:] = temp

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

batch_size = 21



y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#print np.mean(x_train,1)
#print np.mean(x_test,1)


x_train /= np.max(x_train)
x_test /= np.max(x_test)

x_test = torch.from_numpy(x_test).float()
x_train = torch.from_numpy(x_train).float()
y_test = torch.from_numpy(y_test).float()
y_train = torch.from_numpy(y_train).float()



test = torch.utils.data.TensorDataset(x_test, y_test)
train = torch.utils.data.TensorDataset(x_train, y_train)


trainloader = torch.utils.data.DataLoader(dataset = train, batch_size = batch_size, shuffle = True)

testloader = torch.utils.data.DataLoader(dataset = test, batch_size = batch_size)


print "Working on GPU" if torch.cuda.is_available() else "cpu"


def augment_data(image):
	value = random.randint(0,300)
	if value > 250:
		if random.random() < 0.6:
			value = value / 2.0
	op = random.random()
	if op < 0.5:
		image = image * (1 - value/1000.0)
	else:
		image = image * (1 + value/1000.0)
	return image

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.fc1 = nn.Linear(696,1024)
		self.bn1 = nn.BatchNorm1d(1024)
		self.fc2 = nn.Linear(1024,1024)
		self.bn2 = nn.BatchNorm1d(1024)
		self.fc3 = nn.Linear(1024,1024)
		self.bn3 = nn.BatchNorm1d(1024)
		self.fc4 = nn.Linear(1024,4)

	def forward(self, x):
		#print x.size()
		x = augment_data(x)


		x = F.leaky_relu(self.fc1(x))
		#print x.size()

		x = self.bn1(x)
		x = F.dropout(x,0.25)
		#print x.size()
		x = F.leaky_relu(self.fc2(x))
		x = self.bn2(x)
		x = F.dropout(x,0.25)
		#print x.size()

		x = F.leaky_relu(self.fc3(x))
		x = self.bn3(x)
		x = F.dropout(x,0.25)
		#print x.size()

		x = self.fc4(x)

		return x

model = Model()

#model = torch.load('mlp_region_subj1_model.pt')


model = model.cuda()

#print  x_test.size()
#print x_train.size()


#print torch.max(y_train, 1)[1]

def train(epochs, model, trainloader, testloader, lr = 0.001,lower_lr_at = None, lower_lr_rate = 1):
	train_losses = []
	train_acc = []
	test_losses = []
	test_acc = []
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = lr)
	for epoch in range(epochs): # try 200 epoch maybe at 150 lr/3
		if lower_lr_at != None:
			if epoch == lower_lr_at - 1:
				optimizer = optim.Adam(model.parameters(), lr = lr/float(lower_lr_at))
		loss_tmp = 0.0
		correct = 0
		total = 0
		loss_test = 0.0
		for i, (inputs, labels) in enumerate(trainloader):
			inputs, labels = inputs.cuda(), labels.cuda()
			optimizer.zero_grad()
			outputs = model(inputs)
			labels = torch.max(labels, 1)[1]
			loss = criterion(outputs, labels)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum()
			loss.backward()
			optimizer.step()
			loss_tmp += loss.item()
			if i  == 10 - 1:
				print "Epoch: "+str(epoch + 1)+". Train loss is "+ str(loss_tmp/10) + ". Train accuracy is " + str(100*float(correct)/total)
				train_losses.append(loss_tmp / 10)
				train_acc.append(100 * correct/total)
				loss_tmp = 0.0
			# validation
		correct = 0
		total = 0
		with torch.no_grad():
			for inputs, labels in testloader:
				inputs, labels = inputs.cuda(), labels.cuda()
				outputs = model(inputs)
				loss = criterion(outputs, torch.max(labels, 1)[1])
				loss_test += loss.item()
				_, predicted = torch.max(outputs.data, 1)
				labels = torch.max(labels, 1)[1]
				total += labels.size(0)
				correct += (predicted == labels).sum()	
			print "Test loss is "+str(loss_test/10)+ ". Test accuracy is " + str(100.0 * float(correct) / total)
			train_acc.append( float(correct) / total)
			test_losses.append(loss_test/10)
			loss_test = 0.0
	return ((train_losses, train_acc),(test_losses, test_acc))


def test(model, testloader):
	correct = 0
	total = 0
	loss_test = 0.0
	criterion = nn.CrossEntropyLoss()
	with torch.no_grad():
		for (inputs, labels) in testloader:
			inputs, labels = inputs.cuda(), labels.cuda()
			outputs = model(inputs)
			#labels = torch.max(labels, 1)[1]
			loss = criterion(outputs, labels)
			loss_test += loss.item()
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum()
	print "Test loss is "+str(loss_test/10)+ ". Test accuracy is " + str(100.0 * float(correct) / total)
	return (loss_test, float(correct)/total)


((train_loss, train_acc),(test_loss, test_acc)) = train(5000, model,trainloader, testloader, lr = 0.1e-10)

#test(model,testloader)

np.savez('mlp_region_subj1_values.npz', name1=train_loss, name2=train_acc, name3 = test_loss, name4 = test_acc)

torch.save(model, 'mlp_region_subj1_model.pt')
