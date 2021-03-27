# CountLoops
NN to count the number of loops in an RGB image of some text.

### trainer

Train neural network to estimate the number of enclosed spaces in a given text image
To Call:
python ./trainer.py

Args:
*    epoch = 100:        [0-9]+
*    batch size = 1000:  b[0-9]+
*    one-hot = FALSE:    hot | [^(hot)]
*    num_workers = 4:    nw[0-9]+
*    output = False:     o[+]? | [^o]
*    learn rate = 0.01:    lr0\.[0-9]+
*    model = "./loops_counter_net.pth": \./.+\.pth
*    iterations = False: i[0-9]\* \./.+
        Number is how many epochs pass between each save
        Folder is the location of the save

### tester

Test neural network to estimate the number of enclosed spaces in a given text image
To Call:
python ./tester.py

Args:
*    batch size = 1000:  b[0-9]+
*    one-hot = FALSE:    hot | [^(hot)]
*    num_workers = 4:    nw[0-9]+
*    output = False:     o | [^o]
*    net size = 12:      [0-9]+
*    model = "./loops_counter_net.pth": \./.+\.pth

### data/download
Download images to image folder using data in csv file

To Call:
python ./data/download.py

Args:
*    test = False:       t | [^t]
*    display = False:    d | [^d]
*    append = False:     + | [^+]

### data/generator
Generate a given number of dummy images
Generate corresponding csv file with number of loops

To Call:
python ./data/generator.py

Args:
*    n = 100:            [0-9]+
*    test = False:       t | [^t]
*    display = False:    d | [^d]
*    append = False:     + | [^+]

### definitions
Stores Dataset and Transform declarations
Imported into trainer and tester
