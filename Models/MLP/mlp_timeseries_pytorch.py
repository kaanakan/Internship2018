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
os.environ["CUDA_VISIBLE_DEVICES"]="0"


num_classes = 4
batch_size = 210


trnsfm = transforms.Compose([transforms.ToTensor()])

#x_test = scipy.io.loadmat('x_test.mat')['x_test']
#x_train = scipy.io.loadmat('x_train.mat')['x_train']
#x_test = scipy.io.loadmat('teData_mean.mat')['teData_mean']
#x_train = scipy.io.loadmat('trData_mean.mat')['trData_mean']
x_test = scipy.io.loadmat('teData_raw.mat')['teData_raw']
x_train = scipy.io.loadmat('trData_raw.mat')['trData_raw']
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

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#x_train /= np.max(x_train)
x_test /= np.max(x_test)

x_test = torch.from_numpy(x_test).float()
x_train = torch.from_numpy(x_train).float()
y_test = torch.from_numpy(y_test).float()
y_train = torch.from_numpy(y_train).float()



#x_test, x_train, y_test, y_train = x_test.cuda(), x_train.cuda(), y_test.cuda(), y_train.cuda()
"""

x_train1 = x_train.clone() + Variable(torch.randn(x_train.size()).abs_().cuda() * 0.02)

x_train2 = x_train.clone() + Variable(torch.randn(x_train.size()).abs_().cuda() * 0.04)

x_train3 = x_train.clone() + Variable(torch.randn(x_train.size()).abs_().cuda() * 0.06)

x_train4 = x_train.clone() + Variable(torch.randn(x_train.size()).abs_().cuda() * 0.08)

x_train5 = x_train.clone() + Variable(torch.randn(x_train.size()).abs_().cuda() * 0.1)

x_train6 = x_train.clone() - Variable(torch.randn(x_train.size()).abs_().cuda() * 0.02)

x_train7 = x_train.clone() - Variable(torch.randn(x_train.size()).abs_().cuda() * 0.04)

x_train8 = x_train.clone() - Variable(torch.randn(x_train.size()).abs_().cuda() * 0.06)

x_train9 = x_train.clone() - Variable(torch.randn(x_train.size()).abs_().cuda() * 0.08)

"""

x_train1 = x_train.clone() * 1.002

x_train2 = x_train.clone() * 1.004

x_train3 = x_train.clone() * 1.006

x_train4 = x_train.clone() * 1.008

x_train5 = x_train.clone() * 0.998

x_train6 = x_train.clone() * 0.996

x_train7 = x_train.clone() * 0.994

x_train8 = x_train.clone() * 0.992

x_train9 = x_train.clone() * 1.001



x_train /= torch.max(x_train)
x_train1 /= torch.max(x_train1)
x_train2 /= torch.max(x_train2)
x_train3 /= torch.max(x_train3)
x_train4 /= torch.max(x_train4)

x_train5.abs_()
x_train6.abs_()
x_train7.abs_()
x_train8.abs_()
x_train9.abs_()


x_train5 /= torch.max(x_train5)
x_train6 /= torch.max(x_train6)
x_train7 /= torch.max(x_train7)
x_train8 /= torch.max(x_train8)
x_train9 /= torch.max(x_train9)

y_train1 = y_train.clone()
y_train2 = y_train.clone()
y_train3 = y_train.clone()
y_train4 = y_train.clone()

y_train5 = y_train.clone()
y_train6 = y_train.clone()
y_train7 = y_train.clone()
y_train8 = y_train.clone()
y_train9 = y_train.clone()


#y_train1, y_train2, y_train3, y_train4 = y_train1.cuda(), y_train2.cuda(), y_train3.cuda(), y_train4.cuda()



new_x_train = torch.cat((x_train,x_train1, x_train2, x_train3, x_train4,
					x_train5,x_train6, x_train7, x_train8, x_train9),0)

new_y_train = torch.cat((y_train,y_train1, y_train2, y_train3, y_train4,
					y_train5,y_train6, y_train7, y_train8, y_train9),0)


train = torch.utils.data.TensorDataset(new_x_train, new_y_train)
test = torch.utils.data.TensorDataset(x_test, y_test)


trainloader = torch.utils.data.DataLoader(dataset = train, batch_size = 210, shuffle = True)
testloader = torch.utils.data.DataLoader(dataset = test, batch_size = 210)



print "cuda" if torch.cuda.is_available() else "cpu"





class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(137502,1024)
		self.bn1 = nn.BatchNorm1d(1024)
		self.fc2 = nn.Linear(1024,512)
		self.bn2 = nn.BatchNorm1d(512)

		self.fc3 = nn.Linear(512,256)
		self.bn3 = nn.BatchNorm1d(256)
		self.fc4 = nn.Linear(256,128)
		self.bn4 = nn.BatchNorm1d(128)

		self.fc5 = nn.Linear(128,32)
		self.bn5 = nn.BatchNorm1d(32)

		self.fc6 = nn.Linear(32,4)
	

	
	def forward(self, x):
		#batch normalizations after relu
		x = F.leaky_relu(self.fc1(x))
		x = self.bn1(x)

		x = F.dropout(x,0.3)
		x = F.leaky_relu(self.fc2(x))
		x = self.bn2(x)

		x = F.dropout(x,0.3)
		x = F.leaky_relu(self.fc3(x))
		x = self.bn3(x)

		x = F.dropout(x,0.3)

		x = F.leaky_relu(self.fc4(x))
		x = self.bn4(x)
		x = F.dropout(x,0.3)

		x = F.leaky_relu(self.fc5(x))
		x = self.bn5(x)
		x = F.dropout(x,0.3)

		x = self.fc6(x)
		return x
		"""
	def forward(self, x):
		#batch normalizations before relu
		x = self.fc1(x)e
		x = self.bn1(x)
		x = F.relu(x)
		x = F.dropout(x,0.4)
		x = self.fc2(x)
		x = self.bn2(x)
		x = F.relu(x)
		x = F.dropout(x,0.4)
		x = self.fc3(x)
		x = self.bn3(x)
		x = F.relu(x)
		x = F.dropout(x,0.4)
		x = F.relu(self.fc4(x))
		x = F.dropout(x,0.4)
		x = self.fc5(x)
		return x
		"""
	def name(self):
		return "Net"


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
			if i == len(trainloader) - 1:
				print "Epoch: "+str(epoch + 1)+". Train loss is "+ str(loss_tmp/len(trainloader)) + ". Train accuracy is " + str(100*float(correct)/total)
				train_losses.append(loss_tmp / len(trainloader))
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
			print "Test loss is "+str(loss_test/5)+ ". Test accuracy is " + str(100 * float(correct) / total)
			train_acc.append( float(correct) / total)
			test_losses.append(loss_test/5)
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
	print "Test loss is "+str(loss_test/15)+ ". Test accuracy is " + str(100 * float(correct) / total)
	return (loss_test, float(correct)/total)


model = Net()

#model = torch.load('kaan_mlp_pytorch_last.pt')

model = model.cuda()

print(str(model))





print len(trainloader)
print('==>>> total trainning batch number: {}'.format(len(trainloader)))
print('==>>> total testing batch number: {}'.format(len(testloader)))

test(model,testloader)


((train_loss, train_acc),(test_loss, test_acc)) = train(3000,model,trainloader, testloader, lr = 0.001, lower_lr_at = 300, lower_lr_rate = 2)

#test_loss, test_acc = test(model,testloader)

np.savez('mlp_values_graph.npz', name1=train_loss, name2=train_acc, name3 = test_loss, name4 = test_acc)



torch.save(model, 'kaan_mlp_pytorch_last1.pt')
