import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim



import numpy as np
import scipy.io


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="3"


num_classes = 4
batch_size = 210

trnsfm = transforms.Compose([transforms.ToTensor()])


x_test = scipy.io.loadmat('teData_raw_cnn.mat')['teData_raw_cnn']
x_train = scipy.io.loadmat('trData_raw_cnn.mat')['trData_raw_cnn']
y_test = scipy.io.loadmat('tr_te_labels_4class.mat')['te_labels_four_class']
y_train = scipy.io.loadmat('tr_te_labels_4class.mat')['tr_labels_four_class']

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


y_train[y_train == 1] = 0
y_train[y_train == 3] = 1
y_train[y_train == 5] = 2
y_train[y_train == 7] = 3

y_test[y_test == 1] = 0
y_test[y_test == 3] = 1
y_test[y_test == 5] = 2
y_test[y_test == 7] = 3

print(y_train.shape)

y_train = y_train.reshape(210)
y_test  = y_test .reshape(210)
x_train = x_train.reshape(210,6,48,48,48)
x_test = x_test.reshape(210,6,48,48,48)

print(y_train.shape)

#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_test /= np.max(x_test)



x_test = torch.from_numpy(x_test).float()
x_train = torch.from_numpy(x_train).float()
y_test = torch.from_numpy(y_test).float()
y_train = torch.from_numpy(y_train).float()



#x_test, x_train, y_test, y_train = x_test.cuda(), x_train.cuda(), y_test.cuda(), y_train.cuda()


x_train1 = x_train.clone() + Variable(torch.randn(x_train.size()).abs_() * 0.04)

x_train2 = x_train.clone() + Variable(torch.randn(x_train.size()).abs_() * 0.08)

x_train3 = x_train.clone() - Variable(torch.randn(x_train.size()).abs_() * 0.04)

x_train4 = x_train.clone() - Variable(torch.randn(x_train.size()).abs_() * 0.08)

x_train1.abs_()
x_train2.abs_()
x_train3.abs_()
x_train4.abs_()


"""
#try with these
x_train1 = x_train.clone() + Variable(torch.randn(x_train.size()).cuda() * 0.05)

x_train2 = x_train.clone() + Variable(torch.randn(x_train.size()).cuda() * 0.1)

x_train3 = x_train.clone() + Variable(torch.randn(x_train.size()).cuda() * 0.15)

x_train4 = x_train.clone() + Variable(torch.randn(x_train.size()).cuda() * 0.2)

"""



x_train /= torch.max(x_train)
x_train1 /= torch.max(x_train1)
x_train2 /= torch.max(x_train2)
x_train3 /= torch.max(x_train3)
x_train4 /= torch.max(x_train4)




y_train1 = y_train.clone()
y_train2 = y_train.clone()
y_train3 = y_train.clone()
y_train4 = y_train.clone()

#y_train1, y_train2, y_train3, y_train4 = y_train1.cuda(), y_train2.cuda(), y_train3.cuda(), y_train4.cuda()

new_x_train = torch.cat((x_train,x_train1, x_train2, x_train3, x_train4),0)

new_y_train = torch.cat((y_train,y_train1, y_train2, y_train3, y_train4),0)


print new_x_train.size()
print new_y_train.size()


test = torch.utils.data.TensorDataset(x_test, y_test)
train = torch.utils.data.TensorDataset(new_x_train, new_y_train)


trainloader = torch.utils.data.DataLoader(dataset = train, batch_size = 42, shuffle = True)
#trainloader2 = torch.utils.data.DataLoader(dataset = train2, batch_size = 105, shuffle = True)
#trainloader3 = torch.utils.data.DataLoader(dataset = train3, batch_size = 105, shuffle = True)

testloader = torch.utils.data.DataLoader(dataset = test, batch_size = 42)



print "Working on GPU" if torch.cuda.is_available() else "cpu"




class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		self.conv1 = nn.Conv3d(6,32,5)
		self.bn1 = nn.BatchNorm3d(32)
		self.pool0 = nn.MaxPool3d(2)
		self.conv2 = nn.Conv3d(32,32,3)
		self.bn2 = nn.BatchNorm3d(32)
		self.pool1 = nn.MaxPool3d(2)
		#self.conv3 = nn.Conv3d(32,32,3,stride=2)
		#self.bn0 = nn.BatchNorm3d(32)

		#self.pool2 = nn.MaxPool3d(2)
		self.fc1 = nn.Linear(32*10*10*10, 1024)
		self.bn3 = nn.BatchNorm1d(1024)
		self.fc2 = nn.Linear(1024,256)
		self.bn4 = nn.BatchNorm1d(256)
		self.fc3 = nn.Linear(256,64)
		self.bn5 = nn.BatchNorm1d(64)
		self.fc4 = nn.Linear(64,4)
	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.bn1(x)
		x = F.dropout(x,0.2)

		#print x.size()
		x = self.pool0(x)

		x = F.relu(self.conv2(x))
		x = self.bn2(x)
		x = F.dropout(x,0.2)

		#print x.size()
		x = self.pool1(x)
		#print x.size()
		#x = F.relu(self.conv3(x))
		#x = self.bn0(x)
		#x = F.dropout(x,0.2)

		#print x.size()

		#x = F.relu(self.conv4(x))
		#print x.size()

		#x = F.relu(self.conv5(x))
		#print x.size()

		#x = self.pool2(x)
		#print x.size()

		#x = F.relu(self.conv6(x))
		#print x.size()

		#x = self.pool3(x)
		#print x.size()

		x = x.view(-1,32*10*10*10)
		x = F.relu(self.fc1(x))
		x = self.bn3(x)
		x = F.dropout(x,0.25)

		x = F.relu(self.fc2(x))
		x = self.bn4(x)
		x = F.dropout(x,0.25)

		x = F.relu(self.fc3(x))
		x = self.bn5(x)
		x = F.dropout(x,0.25)

		x = self.fc4(x)
		return x
	def name(self):
		return "ConvNet"


class TestNet(nn.Module):
	def __init__(self):
		super(TestNet, self).__init__()
		self.conv1 = nn.Conv3d(6,32,7)
		self.bn1 = nn.BatchNorm3d(32)
		self.conv2 = nn.Conv3d(32,32,7)
		self.bn2 = nn.BatchNorm3d(32)
		self.pool1 = nn.MaxPool3d(2)

		self.conv3 = nn.Conv3d(32,32,5)
		self.bn3 = nn.BatchNorm3d(32)
		self.conv4 = nn.Conv3d(32,32,7)
		self.bn4 = nn.BatchNorm3d(32)
		#self.pool2 = nn.MaxPool3d(2)

		self.conv5 = nn.Conv3d(32,64,3)

		self.fc1 = nn.Linear(64*3*3*3, 1024)
		self.bn5 = nn.BatchNorm1d(1024)
		self.fc2 = nn.Linear(1024,256)
		self.bn6 = nn.BatchNorm1d(256)
		self.fc3 = nn.Linear(256,64)
		self.bn7 = nn.BatchNorm1d(64)
		self.fc4 = nn.Linear(64,4)

	def forward(self, x):
		x = F.leaky_relu(self.conv1(x))
		x = self.bn1(x)
		x = F.dropout(x,0.2)
		#print(x.size())

		x = self.pool1(x)

		x = F.leaky_relu(self.conv2(x))
		x = self.bn2(x)
		x = F.dropout(x,0.2)
		#print(x.size())

		#x = self.pool1(x)
		#print(x.size())

		x = F.leaky_relu(self.conv3(x))
		x = self.bn3(x)
		x = F.dropout(x,0.2)
		#print(x.size())

		x = F.leaky_relu(self.conv4(x))
		x = self.bn4(x)
		x = F.dropout(x,0.2)
		#print(x.size())

		#x = self.pool2(x)

		x = F.leaky_relu(self.conv5(x))
		#print("KEKOKEKO")
		#print(x.size())

		x = x.view(-1, 64*3*3*3)

		x = F.leaky_relu(self.fc1(x))
		x = self.bn5(x)
		x = F.dropout(x,0.25)

		x = F.leaky_relu(self.fc2(x))
		x = self.bn6(x)
		x = F.dropout(x,0.25)

		x = F.leaky_relu(self.fc3(x))
		x = self.bn7(x)
		x = F.dropout(x,0.25)

		x = self.fc4(x)
		return x



model = ConvNet()




#model = torch.load('convNet_pytorch_63.pt')

model.cuda()

print(str(model))

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001/4) # normalde 0.001

print "Total number of training samples: 1050"

print "Total number of testing samples: 210"


#print len(trainloader)
#print('==>>> total trainning batch number: {}'.format(len(trainloader)))
#print('==>>> total testing batch number: {}'.format(len(testloader)))

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
			print(predicted.shape)
			print(labels.shape)
			correct += (predicted == labels).sum()
			loss.backward()
			optimizer.step()
			loss_tmp += loss.item()
			if i  == 25 - 1:
				print "Epoch: "+str(epoch)+". Train loss is "+ str(loss_tmp/25) + ". Train accuracy is " + str(100*float(correct)/total)
				train_losses.append(loss_tmp / 25)
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
			print "Test loss is "+str(loss_test/25)+ ". Test accuracy is " + str(100 * float(correct) / total)
			train_acc.append( float(correct) / total)
			test_losses.append(loss_test/25)
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
			labels = torch.max(labels, 1)[1]
			loss = criterion(outputs, labels)
			loss_test += loss.item()
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum()
	print "Test loss is "+str(loss_test/25)+ ". Test accuracy is " + str(100 * float(correct) / total)
	return (loss_test, float(correct)/total)

#optimizer = optim.Adam(model.parameters(), lr=0.00001/4) # normalde 0.001


((train_loss, train_acc),(test_loss, test_acc)) = train(3000, model,trainloader, testloader, lr = 0.002, lower_lr_at = 450, lower_lr_rate = 2)

#test_loss, test_acc = test(model,testloader)


print "Saving model as \'convNet_pytorch.pt\'"

np.savez('conv_values_graph.npz', name1=train_loss, name2=train_acc, name3 = test_loss, name4 = test_acc)


torch.save(model, 'convNet_pytorch.pt')
