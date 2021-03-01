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
    nw = 4
    o = False
    hot = False
    target = "./loops_counter_net.pth"
    for arg in args:#Takes in command line arguments; less sugar but more rigorous
        if arg[0] == "o":
            o = True
        elif arg[0] == "b":
            try: b = int(arg[1:])
            except: pass
        elif arg[:2] == "nw":
            try: nw = int(arg[2:])
            except: pass
        elif arg[0] == ".":
            target = arg
        elif arg == "hot":
            hot = True
        else:
            print(f"Argument '%s' ignored" % str(arg))

    print(f"Batch Size: %d" % b)
    print(f"Output Loss: %r" % o)
    if hot:
        print("One-Hot encoding: True")
    print(f"Model file: %s\n" % target)

    print("Loading Network")
    net = Net()
    net.load_state_dict(torch.load(target))
    print("Network Loaded\n")

    print("Loading Dataset")
    loops_dataset = LoopsDataset(csv_file='data/test_dat.csv', root_dir='data/test_images/', transform = transforms.Compose([ToTensor()]))
    testloader = DataLoader(loops_dataset, batch_size=b, shuffle=False, num_workers=nw)
    print("Dataset Loaded\n")

    print("Testing Network")
    correct = 0
    total = 0
    all_predictions, all_loops, all_scores = [],[],[]
    with torch.no_grad():
        for data in testloader:
            images, loops, text = data['image'], data['loops'], data['text']
            outputs = net(images)
            predicted = outputs.data#add the _, if one-hot-encoded

            scores = np.array(predicted)
            predicted = np.array([max([(v,i) for i,v in enumerate(predicted[j])])[1] for j in range(len(predicted))])

            total += loops.size(0)

            loops = np.array(loops)

            correct += (predicted == loops).sum()


            all_predictions += list(predicted)
            if o:
                all_loops += list(loops)
                all_scores += list(scores)

    print('Accuracy of the network on the %d test images: %.2f%%' % (len(pd.read_csv("./data/test_dat.csv")), 100 * correct / total))
    print(f"Average guess: %.2f" % (np.array(all_predictions).mean()))
    print(f"SD of guesses: %.2f" % (np.array(all_predictions).var()**.5))

    if o:
        open(f"./%s_guesses.csv" % target, 'w')
        guess_output = pd.DataFrame(all_loops, columns = ["loops"]).assign(guess = all_predictions)
        scores_output = pd.DataFrame(all_scores, columns = [f"s%d" % i for i in range(21)])
        guess_output.join(scores_output).to_csv(f"./%s_guesses.csv" % target[:-4], index = False)

if __name__ == '__main__':
    main(argv[1:])
