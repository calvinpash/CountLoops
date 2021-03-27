import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from definitions import Net, LoopsDataset, ToTensor
from torchvision import transforms, utils
from pandas import DataFrame, concat, read_csv
import numpy as np
from sys import argv, exit
from PIL import Image
import os
#Create Dataset

def make_folder(addy):
    #Make file structure if missing
    print("Makefolder")
    split_addy = addy.split("/")
    for i in range(len(split_addy)-int("." in split_addy[-1])):
        if not os.path.exists("/".join(split_addy[:i+1])):
            os.mkdir("/".join(split_addy[:i+1]))
            print("/".join(split_addy[:i+1]))

def main(args):
    from os import listdir
    e = 100
    b = 1000
    nw = 4
    append = False
    o = False
    o_append = False
    hot = False
    start_target = "./loops_counter_net.pth"
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
        elif arg[0] == "+":
            append = True
            if "." in arg:
                start_target = arg[arg.index("."):]
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

    if append and not os.path.exists(start_target):
        append = False
        print("Couldn't find target starting model")
    if append and model_num >= 0:
        model_num = len(os.listdir(incr_target))

    print(f"Epoch Count: %d" % e)
    print(f"Batch Size: %d" % b)
    print(f"Output Loss: %r" % o)
    print(f"Output model file: %s" % target)
    if append:
        print(f"Starting with model: %r" % start_target)
    if model_num >= 0:
        print(f"\nIncremental Model saving ON")
        print(f"Saving to: %s" % incr_target)
        print(f"Every %d epochs\n" % incr_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device: %s" % device)
    net = Net()
    if append:
        net.load_state_dict(torch.load(start_target))
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
