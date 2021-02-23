import torch
import torchvision
import pandas as pd
import numpy as np
import os
from skimage import io, transform
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
        image = io.imread(img_name)
        text = self.loops_frame.iloc[idx, 1].replace("%2B","+").replace("%23","#").replace("%25","%").replace("%26","&")
        loops = self.loops_frame.iloc[idx, 2]
        sample = {'image': image, 'loops': loops, 'text': text}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    def __call__(self, sample):
        image, loops, text = sample['image'], sample['loops'], sample['text']
        if type(image[0][0])==np.ndarray:
            image = image[:,:,:3]#get rid of alpha channel
        else:#Image is grayscale
            image = np.array([np.array([np.array([px, px, px]) for px in r]) for r in image])
        # image = image.astype(Double)
        image = image/255#Does all the normalization for me
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'loops': loops,
                'text': text}
        #I include the text in here for debugging and since it's closely tied to the loops
