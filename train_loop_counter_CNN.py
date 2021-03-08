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
from sys import argv, exit
import os
#Create Dataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 8, padding = 4)#3 input channels, 12 output, 8x8 kernel
        self.pool = nn.MaxPool2d(4, 4)#summarizes the most activated presence of a feature
        self.conv2 = nn.Conv2d(6, 16, 8, padding = 4)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 21)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)#6x57x57->6x19x19
        x = self.pool(F.relu(x))
        # print(x.shape)#6x57x57->6x19x19
        x = self.conv2(x)
        # print(x.shape)#16x12x12->16x4x4
        x = self.pool(F.relu(x))
        # print(x.shape)#16x12x12->16x4x4
        x = x.view(-1, self.num_flat_features(x))
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = F.relu(self.fc2(x))
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)
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
    nw = 4
    append = False
    o = False
    o_append = False
    hot = False
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
        elif arg == "hot":
            hot = True
        elif arg[0] == "b":
            try: b = int(arg[1:])
            except: pass
        elif arg[:2] == "nw":
            try: nw = int(arg[2:])
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: %s" % device)
    net = Net()
    net.to(device)
    if hot:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    print("Loading Data")
    # device = torch.device('cuda' if torch.cuda)
    loops_dataset = LoopsDataset(hot=hot, csv_file='data/dat.csv', root_dir='data/images/', transform = transforms.Compose([ToTensor()]))
    dataloader = DataLoader(loops_dataset, batch_size = b, shuffle = True, num_workers = nw)

    if len(loops_dataset) != len(os.listdir("./data/images")):
        print(f"Found %d entries and %d images. Killing script" % (len(loops_dataset), len(os.listdir("./data/images"))))
        exit()

    classes = tuple(range(21))

    #We use MSELoss here because our output is a vector of length 21
    optimizer = optim.SGD(net.parameters(), lr=0.1)#,momentum = 0.9)
    print("Loaded\n")


    if o:
        settings = f"e%db%dnw%d" % (e, b, nw)
        loss_file = f"./loss/%s/%s/%s.csv" % (str(criterion)[:-6],
                                                settings,
                                                target[target.rindex("/")+1:target.rindex(".")])

        #Make file structure if missing
        split_addy = loss_file.split("/")
        for i in range(len(split_addy)-1):
            if not os.path.exists("/".join(split_addy[:i+1])):
                os.mkdir("/".join(split_addy[:i+1]))

        loss_output = []#epoch, batch, loss
        if not os.path.exists(loss_file):
            o_append = False
            open(loss_file,'x')
        elif not o_append:
            print("These settings will overwrite an existing loss output file.")
            overwrite = input("Are you sure? (Y/N): ")
            if not overwrite in "yY":
                o_append = True
                print("Good choice, buddy\n")
        print(f"Outputting loss to: %s\n" % loss_file)

    for epoch in range(e):  # loop over the dataset multiple times
        print(f"Current Epoch: %d" % epoch)
        print("\tBatch\tLoss")
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, loops, text = data['image'].to(device), data['loops'].to(device), data['text']
            print(type(inputs), type(loops), type(text))
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
