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


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


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



y_train = y_train.reshape(210)
y_test  = y_test .reshape(210)
x_train = x_train.reshape(210,6,48,48,48)
x_test = x_test.reshape(210,6,48,48,48)



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


#x_test, x_train, y_test, y_train = x_test.cuda(), x_train.cuda(), y_test.cuda(), y_train.cuda()


x_train1 = x_train.clone() + Variable(torch.randn(x_train.size()).abs_() * 0.015)

x_train2 = x_train.clone() + Variable(torch.randn(x_train.size()).abs_() * 0.03)

x_train3 = x_train.clone() - Variable(torch.randn(x_train.size()).abs_() * 0.015)

x_train4 = x_train.clone() - Variable(torch.randn(x_train.size()).abs_() * 0.03)


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


trainloader = torch.utils.data.DataLoader(dataset = train, batch_size = 21, shuffle = True)
#trainloader2 = torch.utils.data.DataLoader(dataset = train2, batch_size = 105, shuffle = True)
#trainloader3 = torch.utils.data.DataLoader(dataset = train3, batch_size = 105, shuffle = True)

testloader = torch.utils.data.DataLoader(dataset = test, batch_size = 21)



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
		x = augment_data(x)
		
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
		self.fc2 = nn.Linear(1024,512)
		self.bn6 = nn.BatchNorm1d(512)
		self.fc3 = nn.Linear(512,128)
		self.bn7 = nn.BatchNorm1d(128)
		self.fc4 = nn.Linear(128,64)
		self.bn8 = nn.BatchNorm1d(64)
		self.fc5 = nn.Linear(64,4)

	def forward(self, x):

		x = augment_data(x)


		x = F.leaky_relu(self.conv1(x))
		x = self.bn1(x)
		x = F.dropout(x,0.3)
		#print(x.size())

		x = self.pool1(x)

		x = F.leaky_relu(self.conv2(x))
		x = self.bn2(x)
		x = F.dropout(x,0.3)
		#print(x.size())

		#x = self.pool1(x)
		#print(x.size())

		x = F.leaky_relu(self.conv3(x))
		x = self.bn3(x)
		x = F.dropout(x,0.3)
		#print(x.size())

		x = F.leaky_relu(self.conv4(x))
		x = self.bn4(x)
		x = F.dropout(x,0.3)
		#print(x.size())

		#x = self.pool2(x)

		x = F.leaky_relu(self.conv5(x))
		#print("KEKOKEKO")
		#print(x.size())

		x = x.view(-1, 64*3*3*3)

		x = F.leaky_relu(self.fc1(x))
		x = self.bn5(x)
		x = F.dropout(x,0.35)

		x = F.leaky_relu(self.fc2(x))
		x = self.bn6(x)
		x = F.dropout(x,0.35)

		x = F.leaky_relu(self.fc3(x))
		x = self.bn7(x)
		x = F.dropout(x,0.35)

		x = F.leaky_relu(self.fc4(x))
		x = self.bn8(x)
		x = F.dropout(x,0.35)

		x = self.fc5(x)
		return x

class Conv2Net(nn.Module):
	def __init__(self):
		super(Conv2Net, self).__init__()
		self.conv1 = nn.Conv3d(6,32,7)
		self.bn1 = nn.BatchNorm3d(32)


		self.conv2_1 = nn.Conv3d(32,64,7)
		self.bn2_1 = nn.BatchNorm3d(64)
		self.conv2_2 = nn.Conv3d(32,64,7)
		self.bn2_2 = nn.BatchNorm3d(64)


		self.conv3_1 = nn.Conv3d(128,64,5)
		self.bn3_1 = nn.BatchNorm3d(64)
		self.conv3_2 = nn.Conv3d(128,64,5)
		self.bn3_2 = nn.BatchNorm3d(64)

		self.conv3_3 = nn.Conv3d(128,128,3)
		self.bn3_3 = nn.BatchNorm3d(128)


		self.conv4_1 = nn.Conv3d(128,128,3)
		self.bn4_1 = nn.BatchNorm3d(128)
		self.conv4_2 = nn.Conv3d(128,128,3)
		self.bn4_2 = nn.BatchNorm3d(128)
		
		self.conv4_3 = nn.Conv3d(256,512,3)
		self.bn4_3 = nn.BatchNorm3d(512)


		self.pool = nn.MaxPool3d(2)

		self.fc1 = nn.Linear(512 * 5 * 5 * 5, 1024)
		self.bnf1 = nn.BatchNorm1d(1024)

		self.fc2 = nn.Linear(1024, 1024)
		self.bnf2 = nn.BatchNorm1d(1024)

		self.fc3 = nn.Linear(1024, 1024)
		self.bnf3 = nn.BatchNorm1d(1024)

		self.fc4 = nn.Linear(1024, 4)
		
	def forward(self, x):
		x = augment_data(x)

		x = F.leaky_relu(self.conv1(x))
		x = self.bn1(x)
		x = F.dropout(x,0.25)

		out1 = F.leaky_relu(self.conv2_1(x))
		out1 = self.bn2_1(out1)
		out1 = F.dropout(out1, 0.25)

		out2 = F.leaky_relu(self.conv2_2(x))
		out2 = self.bn2_2(out2)
		out2 = F.dropout(out2, 0.25)

		x = torch.cat((out1, out2), 1)

		#x = self.pool(x)

		out1 = F.leaky_relu(self.conv3_1(x))
		out1 = self.bn3_1(out1)
		out1 = F.dropout(out1, 0.25)

		out2 = F.leaky_relu(self.conv3_2(x))
		out2 = self.bn3_2(out2)
		out2 = F.dropout(out2, 0.25)

		x = torch.cat((out1, out2), 1)

		x = self.pool(x)


		x = F.leaky_relu(self.conv3_3(x))
		x = self.bn3_3(x)
		x = F.dropout(x, 0.25)


		#x = self.pool(x)

		out1 = F.leaky_relu(self.conv4_1(x))
		out1 = self.bn4_1(out1)
		out1 = F.dropout(out1, 0.25)

		out2 = F.leaky_relu(self.conv4_2(x))
		out2 = self.bn4_2(out2)
		out2 = F.dropout(out2, 0.25)

		x = torch.cat((out1, out2), 1)

		x = F.leaky_relu(self.conv4_3(x))
		x = self.bn4_3(x)
		x = F.dropout(x, 0.25)

		x = self.pool(x)

		x = x.view(-1, 512 * 5 * 5 * 5)

		x = F.leaky_relu(self.fc1(x))
		x = self.bnf1(x)
		x = F.dropout(x,0.3)

		x = F.leaky_relu(self.fc2(x))
		x = self.bnf2(x)
		x = F.dropout(x,0.3)

		x = F.leaky_relu(self.fc3(x))
		x = self.bnf3(x)
		x = F.dropout(x,0.3)

		x = self.fc4(x)

		return x


model = Conv2Net()



#model = torch.load('convNet_pytorch_augment.pt')

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
			correct += (predicted == labels).sum()
			loss.backward()
			optimizer.step()
			loss_tmp += loss.item()
			if i  == 50 - 1:
				print "Epoch: "+str(epoch + 1)+". Train loss is "+ str(loss_tmp/50) + ". Train accuracy is " + str(100*float(correct)/total)
				train_losses.append(loss_tmp / 50)
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
			print "Test loss is "+str(loss_test/50)+ ". Test accuracy is " + str(100 * float(correct) / total)
			train_acc.append( float(correct) / total)
			test_losses.append(loss_test/50)
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
	print "Test loss is "+str(loss_test/50)+ ". Test accuracy is " + str(100 * float(correct) / total)
	return (loss_test, float(correct)/total)

#optimizer = optim.Adam(model.parameters(), lr=0.00001/4) # normalde 0.001


((train_loss, train_acc),(test_loss, test_acc)) = train(200, model,trainloader, testloader, lr = 0.0003, lower_lr_at = 18, lower_lr_rate = 2.2)

#test_loss, test_acc = test(model,testloader)


print "Saving model as \'conv2Net_pytorch.pt\'"

np.savez('conv2net_values_graph.npz', name1=train_loss, name2=train_acc, name3 = test_loss, name4 = test_acc)


torch.save(model, 'conv2Net_pytorch.pt')