from requests import get
import os
from os import listdir, mkdir
from sys import argv
from pandas import read_csv

def main(args):
    d = False
    t = False
    append = False
    for arg in args:#Takes in command line args
        if arg == "+":
            append = True
        elif arg == "d":
            d = True
        elif arg == "t":
            t = True
        else:
            print(f"Argument '%s' ignored" % str(arg))
    target = "./" + ("test_" if t else "") + "images"
    targetcsv = "./" + ("test_" if t else "") + "dat.csv"
    append = (append and os.path.exists(target))

    offset = 0
    if append:
        #Get current count of image files in folder
        offset += len([i for i in listdir(target) if i.endswith(".png")])
    if not os.path.exists(targetcsv):
        return TypeError("Missing necessary csv file: " + targetcsv)
    if not os.path.exists(target):
        mkdir(target)

    print(f"Downloading %d entries" % (len(read_csv(targetcsv)) - offset))
    print(f"Appending: %r" % append)
    print(f"Target folder: %s\n" % target)

    print("Downloading...")
    if d:
        print("Index\tText")
    with open(targetcsv) as file:
        csv = read_csv(file)
        for row in csv[offset:].iloc:
            #Obviously, this isn't practical for a large scale, but this is a quick and dirty way to get a small set of random images
            url = f"https://dummyimage.com/64.png/%s/%s/&text=?%s" % (row[4], row[3], row[1])
            open(f"%s/%s.png" % (target,row[0]),"wb").write(get(url).content)
            if d:
                print("%s\t%s" % (row[0], row[1]))
    print("Downloading Finished")

main(argv[1:])
