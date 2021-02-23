'''
Download images using data in dat.csv

Call:
python download.py (test?: t for true) (append: + for True)
'''
from requests import get
import os
from os import listdir, mkdir
from sys import argv
import csv

def main(args):
    test = (len(args) > 0 and args[-1] == "t") or (len(args) > 1 and args[-2] == "t")
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

    with open(targetcsv) as file:
        r = csv.reader(file)
        for row in list(r)[offset:]:
            #Obviously, this isn't practical for a large scale, but this is a quick and dirty way to get a small set of random images
            url = f"https://dummyimage.com/64.png/%s/%s/&text=%s" % (row[3], row[4], row[1])
            open(f"%s/%s.png" % (target,row[0]),"wb").write(get(url).content)

main(argv[1:])