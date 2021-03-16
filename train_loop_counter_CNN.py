import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms, utils
from pandas import DataFrame, concat, read_csv
import numpy as np
from sys import argv, exit
from PIL import Image
import os
#Create Dataset

class Net(nn.Module):
    def __init__(self, output = 12):
        super(Net, self).__init__()
        padding1=0
        filter1=3

        self.conv0 = nn.Conv2d(3, 1, 1, padding = 0)#Reduce RGB channels to one value
        self.conv1 = nn.Conv2d(3, 4, filter1, padding = padding1)
        size1=64+2*padding1-(filter1-1)
        #print(size1)
        pool_size=2 #pooling kernel
        pool_stride=2 #pooling size

        self.pool = nn.MaxPool2d(pool_size, pool_stride)#summarizes the most activated presence of a feature

        size2=np.floor((size1 - pool_size)/pool_stride + 1).astype(int)
        #print(size2)

        padding2=0
        filter2=3

        self.conv2 = nn.Conv2d(4, 4, filter2, padding = padding2)
        size3 = size2+2*padding2-(filter2-1)
        #print(size3)

        #self.fc1 = nn.Linear(16 * 4 * 4, 120)
        size4=np.floor((size3 - pool_size)/pool_stride + 1).astype(int)
        #print(f"%d\t%d\t%d\t%d" % (size1, size2, size3, size4))

        padding3 = 0
        filter3 = 3
        self.conv3 = nn.Conv2d(4, 4, filter3, padding = padding3)

        size5 = size4+2*padding3-(filter3-1)
        self.fc1 = nn.Linear(4 * size5 * size5, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80,output)

    def forward(self, x):
        #x = self.conv0(x)
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = self.pool(F.relu(x))
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = self.pool(F.relu(x))
        #print(x.shape)
        x = self.conv3(x)
        #print(x.shape)
        x = x.view(-1, self.num_flat_features(x))
        #print(x.shape)
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

    def get_output_size(self):
        return self.fcs.out_features
    
class LoopsDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, hot = False):
        self.loops_frame = read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.hot = hot

    def __len__(self):
        return len(self.loops_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, str(self.loops_frame.iloc[idx, 0]) + ".png")

        tmp=Image.open(img_name).convert('RGB')

        image = np.zeros([tmp.size[0],tmp.size[1],3])
        image[:,:,0],image[:,:,1],image[:,:,2]=tmp.split()
        #print(image.shape)
        text = self.loops_frame.iloc[idx, 1].replace("%2B","+").replace("%23","#").replace("%25","%").replace("%26","&")

        if self.hot: #If using one-hot encoded list (MSE, . . .)
            loops = np.zeros(21)
            loops[self.loops_frame.iloc[idx, 2]] = 1
        else: #If using label index (CrossEntropy, . . .
            loops = self.loops_frame.iloc[idx, 2]
        sample = {'image': image, 'loops': loops, 'text': text}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    def __call__(self, sample):
        image, loops, text = sample['image'], sample['loops'], sample['text']

        hot = (type(loops) == np.ndarray)
        if image.shape[2] == 4:#if the image has a color channel
            image = image[:,:,:3]#get rid of alpha channel
        elif len(image.shape) == 2:#if the image is grayscale, create color channels
            image = np.array([np.array([np.array([px, px, px]) for px in r]) for r in image])
        image = image/255.#Does all the normalization for me
        #Convert image shape from (64,64,3) to (3,64,64)

        #print(image.shape)
        image = image.transpose((2, 0, 1))
        if hot:
            return {'image': torch.from_numpy(image).float(),
                    'loops': torch.from_numpy(loops).float(),
                    'text': text}
        else:
            return {'image': torch.from_numpy(image).float(),
                    'loops': loops,
                    'text': text}

def make_folder(addy):
    #Make file structure if missing
    print("Makefolder")
    split_addy = addy.split("/")
    for i in range(len(split_addy)-int("." in split_addy[-1])):
        if not os.path.exists("/".join(split_addy[:i+1])):
            os.mkdir("/".join(split_addy[:i+1]))
            print("/".join(split_addy[:i+1]))

def main(args):
    e = 100
    b = 1000
    nw = 4
    append = False
    o = False
    o_append = False
    hot = False
    target = "./loops_counter_net.pth"
    data_file='data/dat.csv'
    data_dir='data/images/'
    model_num = -1
    incr_size = 25
    incr_target = "./incr_model"
    lr = 0.01
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
        elif arg[0] == "i":
            model_num = 0
            if len(arg) > 1:
                for i in range(4):
                    try: incr_size = int(arg[1:1+i])
                    except: continue
            if "." in arg:
                incr_target = arg[arg.index("."):]
                make_folder(incr_target)
        elif len(arg) > 1 and arg[:2] == "lr":
             try: lr = float(arg[3:])
             except: pass
        else:
            print(f"Argument '%s' ignored" % str(arg))

    if not os.path.exists(target):
        append = False

    print(f"Epoch Count: %d" % e)
    print(f"Batch Size: %d" % b)
    print(f"Output Loss: %r" % o)
    print(f"Model file: %s" % target)
    print(f"Append to model: %r" % append)
    if model_num >= 0:
        print(f"\nIncremental Model saving ON")
        print(f"Saving to: %s" % incr_target)
        print(f"Every %d epochs\n" % incr_size)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device: %s" % device)
    net = Net()
    if append and os.path.exists(target):
        net.load_state_dict(torch.load(target))
        if hot:
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        vals=np.ones(12)
        vals=vals/np.linalg.norm(vals)
        weights=torch.FloatTensor(vals).to(device)
        if hot:
            criterion = nn.MSELoss(weight = weights)
        else:
            criterion = nn.CrossEntropyLoss(weight = weights)

    net.to(device)
    print("Loading Data")
    # device = torch.device('cuda' if torch.cuda)
    loops_dataset = LoopsDataset(hot=hot, csv_file=data_file, root_dir=data_dir, transform = transforms.Compose([ToTensor()]))
    dataloader = DataLoader(loops_dataset, batch_size = b, num_workers = nw)

    if len(loops_dataset) != len(os.listdir(data_dir)):
        print(f"Found %d entries and %d images. Killing script" % (len(loops_dataset), len(os.listdir(data_dir))))
        exit()
    if model_num >= 0 and incr_size > e:
        print("Incremental save size greater than epoch.")
        exit()

    classes = tuple(range(21))

    #We use MSELoss here because our output is a vector of length 21
    optimizer = optim.SGD(net.parameters(), lr=lr)#,momentum = 0.9)
    print("Loaded\n")



    if o:
        settings = f"e%db%dnw%d" % (e, b, nw)
        loss_file = f"./loss/%s/%s/%s.csv" % (str(criterion)[:-6],
                                                settings,
                                                (target[target.rindex("/")+1:target.rindex(".")] if model_num < 0 else incr_target[incr_target.rindex("/")+1:]))

        make_folder(loss_file)

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

    compact_print = (len(loops_dataset)/b > 10)
    if compact_print:
        print("Printing compactly. %d Batches per Epoch" % np.floor(len(loops_dataset)/b))
        test_dataset = LoopsDataset(csv_file='data/test_dat.csv', root_dir='data/test_images/', transform = transforms.Compose([ToTensor()]))
        testloader = DataLoader(test_dataset, batch_size=b, shuffle=False, num_workers=nw)
        test_acc = []

    
    for epoch in range(e):  # loop over the dataset multiple times
        print(f"Current Epoch: %d" % epoch)
        running_loss = 0.0
        if compact_print:
            losses = []
            acces = []
        else:
            print("\tBatch\tLoss\tAccuracy\t")

        for i, data in enumerate(dataloader, 0):
            inputs, loops, text = data['image'].to(device), data['loops'].to(device), data['text']
            # print(type(inputs), type(loops), type(text))
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, loops)

            _, pred = outputs.max(dim=1)
            correct = int(pred.eq(loops).sum().item())
            acc = correct / int(loops.size()[0])
        
            if compact_print:
                losses.append(loss.sum().item())
                acces.append(acc)
            else:
                print(f"\t%d\t%.4f\t%.3f" % (i, loss.sum().item(), acc))
                if o:
                    loss_output.append([epoch, i, loss.sum().item()])
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # if i % 200 == 199:    # print every 200 batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 200))
            #     running_loss = 0.0
        if model_num >= 0 and epoch % incr_size == 0:
            torch.save(net.state_dict(), f"%s/%d.pth" % (incr_target, model_num))
            model_num += 1
            all_predictions = []

            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, loops = data['image'].to(device), data['loops']
                    outputs = net(images)
                    predicted = outputs.data                    
                    
                    predicted = np.array([max([(v,i) for i,v in enumerate(predicted[j])])[1] for j in range(len(predicted))])
                    
                    total += loops.size(0)                   
                    loops = np.array(loops)
                    correct += (predicted == loops).sum()
                    all_predictions += list(predicted)
            print('Accuracy on test: %.2f%%' % (100 * correct / total))
            print(f"Avg guess: %.2f" % (np.array(all_predictions).mean()))
            print(f"SD of guesses: %.2f" % (np.array(all_predictions).var()**.5))
            test_acc.append(correct/total)


        if compact_print:
            print("\tMean\tSD")
            lm, lsd = (np.array(losses).mean(), np.array(losses).var()**.5)
            am, asd = (np.array(acces).mean(), np.array(acces).var()**.5)
            print(f"Loss\t%.4f\t%.4f" % (lm, lsd))
            print(f"Acc\t%.4f\t%.4f" % (am, asd))
            if o:
                loss_output.append([epoch, lm, am, lsd, asd])  
        print('Finished Training')

    if o:
        if o_append:
            df = read_csv(loss_file)
        elif compact_print:
            df = DataFrame(columns = ["epoch","loss","acc","loss_sd","acc_sd", "test_acc", "train_number"])
            loss_output = [loss_output[i] + [test_acc[i//incr_size]] for i in range(len(loss_output))]
        else:
            df = DataFrame(columns = ["epoch","batch","loss", "train_number"])

        df_output = DataFrame(loss_output, columns = (["epoch","loss","acc","loss_sd","acc_sd", "test_acc"] if compact_print else ["epoch","batch","loss"]))
        df_output = df_output.assign(train_number = np.full(len(loss_output), len(df.train_number.unique())))
        df = concat([df, df_output])
        df.to_csv(loss_file, index = False)

    torch.save(net.state_dict(), target)

if __name__ == '__main__':
    main(argv[1:])
