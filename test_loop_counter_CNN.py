'''
Test neural network to estimate the number of enclosed spaces in a given text image
To Call:
python test_loop_counter_CNN.py (batch size = 1000: b[0-9]+)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from train_loop_counter_CNN import Net
from data.dataset_definitions import LoopsDataset, ToTensor
from sys import argv, exit
import numpy as np
import pandas as pd

def main(args):
    b = 1000
    for arg in args:#Takes in command line arguments; less sugar but more rigorous
        if arg[0] == "b":
            try: b = int(arg[1:])
            except: pass
        else:
            print(f"Argument '%s' ignored" % str(arg))
    print(f"Batch Size: %d\n" % b)

    print("Loading Network")
    net = Net()
    net.load_state_dict(torch.load('./loops_counter_net.pth'))
    print("Network Loaded\n")

    print("Loading Dataset")
    loops_dataset = LoopsDataset(csv_file='data/test_dat.csv', root_dir='data/test_images/', transform = transforms.Compose([ToTensor()]))
    testloader = DataLoader(loops_dataset, batch_size=1000, shuffle=False, num_workers=2)
    print("Dataset Loaded\n")

    print("Testing Network")
    correct = 0
    total = 0
    all_predictions, all_loops = [], []
    with torch.no_grad():
        for data in testloader:
            images, loops, text = data['image'], data['loops'], data['text']
            outputs = net(images)

            predicted = outputs.data#add the _, if one-hot-encoded
            predicted = np.array([max([(v,i) for i,v in enumerate(predicted[j])])[1] for j in range(len(predicted))])

            total += loops.size(0)

            #For non-one-hot-encoded loss functions (CrossEntropy, . . .)
            loops = np.array(loops)
            #For one-hot-encoded loss functions (MSE, . . .)
            # loops = np.array([max([(v,i) for i,v in enumerate(loops[j])])[1] for j in range(len(loops))])

            correct += (predicted == loops).sum()

            all_predictions += list(predicted)
            all_loops += list(loops)

    print('Accuracy of the network on the %d test images: %.2f%%' % (len(pd.read_csv("./data/test_dat.csv")), 100 * correct / total))
    print(f"Average guess: %.2f" % (np.array(all_predictions).mean()))
    print(f"SD of guesses: %.2f" % (np.array(all_predictions).var()**.5))

    open("./guesses.csv", 'w')
    pd.DataFrame(all_loops, columns = ["loops"]).assign(guess = all_predictions).to_csv("./guesses.csv", index = False)

if __name__ == '__main__':
    main(argv[1:])
