'''
Train neural network to estimate the number of enclosed spaces in a given text image
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, utils
from data.dataset_definitions import LoopsDataset, ToTensor
#Create Dataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 8)#3 input channels, 12 output, 8x8 kernel
        self.pool = nn.MaxPool2d(3, 3)#summarizes the most activated presence of a feature
        #This
        self.conv2 = nn.Conv2d(6, 16, 8)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def main():
    loops_dataset = LoopsDataset(csv_file='data/dat.csv', root_dir='data/images/', transform = transforms.Compose([ToTensor()]))
    dataloader = DataLoader(loops_dataset, batch_size = 4, shuffle = True, num_workers = 2)
    classes = tuple(range(21))

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1):  # loop over the dataset multiple times
        count = 0
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, loops]
            inputs, loops, text = data['image'], data['loops'], data['text']
            # print(text[0] + "|")
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, loops)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    print('Finished Training')

    torch.save(net.state_dict(), './loops_counter_net.pth')

if __name__ == '__main__':
    main()
