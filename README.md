# CountLoops
NN to count the number of loops in an RGB image of some text.

###train_loop_counter_CNN

Train neural network to estimate the number of enclosed spaces in a given text image
To Call:
python ./train_loop_counter_CNN.py

Args:
    epoch = 100:        [0-9]+
    batch size = 1000:  b[0-9]+
    one-hot = FALSE:    hot | [^(hot)]
    num_workers = 4:    nw[0-9]+
    output = False:     o[+]? | [^o]
    model = "./loops_counter_net.pth": \./.+\.pth

###test_loop_counter_CNN

Test neural network to estimate the number of enclosed spaces in a given text image
To Call:
python ./train_loop_counter_CNN.py

Args:
    batch size = 1000:  b[0-9]+
    one-hot = FALSE:    hot | [^(hot)]
    num_workers = 4:    nw[0-9]+
    output = False:     o | [^o]
    model = "./loops_counter_net.pth": \./.+\.pth

###data/download
Download images to image folder using data in csv file

To Call:
python ./data/download.py

Args:
    test = False:       t | [^t]
    display = False:    d | [^d]
    append = False:     + | [^+]

###data/generator
Generate a given number of dummy images
Generate corresponding csv file with number of loops

To Call:
python ./data/generator.py

Args:
    n = 100:            [0-9]+
    test = False:       t | [^t]
    display = False:    d | [^d]
    append = False:     + | [^+]

###data/dataset_definitions
Stores Dataset and Transform declarations
Imported into trainer and tester
