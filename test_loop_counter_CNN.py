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
from sys import argv

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
    loops_dataset = LoopsDataset(csv_file='data/dat.csv', root_dir='data/images/', transform = transforms.Compose([ToTensor()]))
    testloader = DataLoader(loops_dataset, batch_size=1000, shuffle=False, num_workers=2)
    print("Dataset Loaded\n")

    print("Testing Network")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, loops, text = data['image'], data['loops'], data['text']
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += loops.size(0)
            correct += (predicted == loops).sum().item()

    print('Accuracy of the network on the 1000 test images: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    main(argv[1:])
