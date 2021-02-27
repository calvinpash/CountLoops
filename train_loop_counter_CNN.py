'''
Train neural network to estimate the number of enclosed spaces in a given text image
To Call:
python train_loop_counter_CNN.py (epoch = 100: [0-9]+) (batch size = 1000: b[0-9]+) (output? = False: o[+]?|[^o]) (model = "./loops_counter_net.pth": \./.+\.pth)(append?: +|[^+] )
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, utils
from data.dataset_definitions import LoopsDataset, ToTensor
from pandas import DataFrame, concat, read_csv
import numpy as np
from sys import argv
import os
from sys import exit #helpful for troubleshooting
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
    o = False
    o_append = False
    target = "./loops_counter_net.pth"
    for arg in args:#Takes in command line arguments; less sugar but more rigorous
        try: arg = int(arg)
        except: pass
        if type(arg) == int:
            e = arg
        elif arg == "+":
            append = True
        elif arg[0] == "o":
            o = True
            if len(arg) > 1 and arg[1] == "+":
                o_append = True
        elif arg[0] == "b":
            try: b = int(arg[1:])
            except: pass
        elif arg[0] == ".":
            target = arg
        else:
            print(f"Argument '%s' ignored" % str(arg))

    if not os.path.exists(target):
        append = False

    print(f"Epoch Count: %d" % e)
    print(f"Batch Size: %d" % b)
    print(f"Output Loss: %r" % o)
    print(f"Model file: %s" % target)
    # print(f"Append to model: %r" % append)

    net = Net()
    criterion = nn.MSELoss()
    print("Loading Data")
    # device = torch.device('cuda' if torch.cuda)
    loops_dataset = LoopsDataset(csv_file='data/dat.csv', root_dir='data/images/', transform = transforms.Compose([ToTensor()]))
    dataloader = DataLoader(loops_dataset, batch_size = b, shuffle = True, num_workers = 4)

    classes = tuple(range(21))

    #We use MSELoss here because our output is a vector of length 21
    optimizer = optim.SGD(net.parameters(), lr=0.1)#,momentum = 0.9)
    print("Loaded\n")


    if o:
        if not os.path.exists(f"./loss/%s" % str(criterion)[:-6]):
            os.mkdir(f"loss/%s" % str(criterion)[:-6])
        settings = f"e%db%dnw%d" % (e, b, dataloader.num_workers)
        loss_file = f"./loss/%s/%s_%s.csv" % (str(criterion)[:-6], target[target.rindex("/")+1:target.rindex(".")], settings)
        loss_output = []#epoch, batch, loss
        if not os.path.exists(loss_file):
            o_append = False
        elif not o_append:
            print("These settings will overwrite an existing loss output file.")
            overwrite = input("Are you sure? (Y/N): ")
            if not overwrite in "yY":
                o_append = True
                print("Good choice, buddy\n")
        print(f"Outputting loss to: %s\n" % loss_file)

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
            if o:
                loss_output.append([epoch, i, loss.sum().item()])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # if i % 200 == 199:    # print every 200 batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 200))
            #     running_loss = 0.0

    print('Finished Training')

    if o:
        if o_append:
            df = read_csv(loss_file)
        else:
            df = DataFrame(columns = ["epoch","batch","loss", "train_number"])
    # with open(target, "w", newline = '') as csvfile:
        df_output = DataFrame(loss_output, columns = ["epoch","batch","loss"])
        df_output = df_output.assign(train_number = np.full(len(loss_output), len(df.train_number.unique())))
        df = concat([df, df_output])
        df.to_csv(loss_file, index = False)
    torch.save(net.state_dict(), target)

if __name__ == '__main__':
    main(argv[1:])
