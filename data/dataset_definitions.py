import torch
import torchvision
import pandas as pd
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class LoopsDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.loops_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.loops_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, str(self.loops_frame.iloc[idx, 0]) + ".png")
        image = np.array(Image.open(img_name))
        text = self.loops_frame.iloc[idx, 1].replace("%2B","+").replace("%23","#").replace("%25","%").replace("%26","&")

        #If using one-hot encoded list (MSE, . . .)
        # loops = np.zeros(21)
        # loops[self.loops_frame.iloc[idx, 2]] = 1

        #If using label index (CrossEntropy, . . .
        loops = self.loops_frame.iloc[idx, 2]
        sample = {'image': image, 'loops': loops, 'text': text}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    def __call__(self, sample):
        image, loops, text = sample['image'], sample['loops'], sample['text']

        if len(list(image.shape)) == 3:#if the image has a color channel
            image = image[:,:,:3]#get rid of alpha channel
        else:#if the image is grayscale, create color channels
            image = np.array([np.array([np.array([px, px, px]) for px in r]) for r in image])
        image = image/255#Does all the normalization for me
        #Convert image shape from (64,64,3) to (3,64,64)
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                #For one-hot-encoded
                # 'loops': torch.from_numpy(loops).float(),
                #For non-one-hot-encoded
                'loops':loops,
                'text': text}
        #I include the text in here for debugging and since it's closely tied to the loops
