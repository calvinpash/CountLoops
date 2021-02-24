'''
Train neural network to estimate the number of enclosed spaces in a given text image
To Call:
python train_loop_counter_CNN.py (epoch = 100: [0-9]+) (batch size = 1000: b[0-9]+) (append?: +|(^$) )
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, utils
from data.dataset_definitions import LoopsDataset, ToTensor
from sys import argv
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
        self.fc3 = nn.Linear(84, 21)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  #all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def main(args):
    e = 100
    b = 1000
    append = False
    for arg in args:#Takes in command line arguments; less sugar but more rigorous
        try: arg = int(arg)
        except: pass
        if type(arg) == int:
            e = arg
        elif arg == "+":
            append = True
        elif arg[0] == "b":
            try: b = int(arg[1:])
            except: pass
        else:
            print(f"Argument '%s' ignored" % str(arg))
    print(f"Epoch Count: %d" % e)
    print(f"Batch Size: %d" % b)
    # print(f"Append to model: %r" % append)


    print("Loading Data")
    # device = torch.device('cuda' if torch.cuda)
    loops_dataset = LoopsDataset(csv_file='data/dat.csv', root_dir='data/images/', transform = transforms.Compose([ToTensor()]))
    dataloader = DataLoader(loops_dataset, batch_size = b, shuffle = True, num_workers = 4)
    classes = tuple(range(21))

    net = Net()
    criterion = nn.MSELoss()
    #We use MSELoss here because our output is a vector of length 21
    optimizer = optim.SGD(net.parameters(), lr=0.1)#,momentum = 0.9)
    print("Loaded\n")
    for epoch in range(e):  # loop over the dataset multiple times
        print(f"Current Epoch: %d" % epoch)
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            print("\tBatch\tLoss")
            inputs, loops, text = data['image'], data['loops'], data['text']

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, loops)

            print(f"\t%d\t%f" % (i, loss.sum().item()))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # if i % 200 == 199:    # print every 200 batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 200))
            #     running_loss = 0.0

    print('Finished Training')

    torch.save(net.state_dict(), './loops_counter_net.pth')

if __name__ == '__main__':
    main(argv[1:])
