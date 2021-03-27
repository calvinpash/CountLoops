import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pandas import DataFrame, concat, read_csv
import numpy as np
from PIL import Image
import os

class Net(nn.Module):
    def __init__(self, output = 12):
        super(Net, self).__init__()

        padding1=1
        filter1=3
        output1 = 32

        self.conv1 = nn.Conv2d(3, output1, filter1, padding = padding1)

        size1=64+2*padding1-(filter1-1)

        pool_size=2 #pooling kernel
        pool_stride=2 #pooling size

        self.pool = nn.MaxPool2d(pool_size, pool_stride)#summarizes the most activated presence of a feature

        size2=np.floor((size1 - pool_size)/pool_stride + 1).astype(int)

        padding2=2
        filter2=5
        output2=32

        self.conv2 = nn.Conv2d(output1, output2, filter2, padding = padding2)

        size3 = size2+2*padding2-(filter2-1)
        size4=np.floor((size3 - pool_size)/pool_stride + 1).astype(int)

        padding3 = 1
        filter3 = 3
        output3 = 32

        self.conv3 = nn.Conv2d(output2, output3, filter3, padding = padding3)

        size5 = size4+2*padding3-(filter3-1)
        size6 = np.floor((size5 - pool_size)/pool_stride + 1).astype(int)

        filter4 = 3
        padding4 = 1
        output4 = 8

        self.conv4 = nn.Conv2d(output3, output4, filter4, padding = padding4)

        size7 = size6+2*padding4-(filter4-1)
        size8 = np.floor((size7 - pool_size)/pool_stride + 1).astype(int)

        self.fc1 = nn.Linear(output4 * size8 * size8, 120)
        self.fc_out = nn.Linear(120,output)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        x = self.conv3(x)
        x = self.pool(F.relu(x))
        x = self.conv4(x)
        x = self.pool(F.relu(x))

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = self.fc_out(x)
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
        text = str(self.loops_frame.iloc[idx, 1]).replace("%2B","+").replace("%23","#").replace("%25","%").replace("%26","&")

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

        image = image.transpose((2, 0, 1))
        if hot:
            return {'image': torch.from_numpy(image).float(),
                    'loops': torch.from_numpy(loops).float(),
                    'text': text}
        else:
            return {'image': torch.from_numpy(image).float(),
                    'loops': loops,
                    'text': text}
