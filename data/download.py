'''
Download images using data in dat.csv

Call:
python download.py (test?: t for true) (display?: d for true) (append: + for True)
'''
from requests import get
import os
from os import listdir, mkdir
from sys import argv
from pandas import read_csv

def main(args):
    disp = (args[-1] == "d" or
        (len(args) > 1 and args[-2] == "d") or
        (len(args) > 2 and args[-3] == "d"))
    test = (args[-1] == "t" or
        (len(args) > 1 and args[-2] == "t") or
        (len(args) > 2 and args[-3] == "t"))
    target = "./" + ("test_" if test else "") + "images"
    targetcsv = "./" + ("test_" if test else "") + "dat.csv"
    append = (len(args) > 0 and (args[-1] == "+"))
    offset = 1#1 because of the first line of the csv
    if append:
        #Get current count of image files in folder
        offset += len([i for i in listdir(target) if i.endswith(".png")])
    if not os.path.exists(targetcsv):
        return TypeError("Missing necessary csv file: " + targetcsv)
    if not os.path.exists(target):
        mkdir(target)


    print(f"Downloading %d entries" % (offset-1))
    print(f"Appending: %r" % append)
    print(f"Target folder: %s\n" % target)

    print("Downloading...\n")
    if disp:
        print("Index\tText")
    with open(targetcsv) as file:
        csv = read_csv(file)
        for row in csv[offset:].iloc:
            #Obviously, this isn't practical for a large scale, but this is a quick and dirty way to get a small set of random images
            url = f"https://dummyimage.com/64.png/%s/%s/&text=%s" % (row[3], row[4], row[1])
            open(f"%s/%s.png" % (target,row[0]),"wb").write(get(url).content)
            if disp:
                print("%s\t%s" % (row[0], row[1]))
    print("Downloading Finished")

main(argv[1:])
