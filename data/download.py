'''
Download images using data in dat.csv

Call:
python download.py (append: + for True)
'''
from requests import get
from os import listdir, mkdir
from sys import argv
import csv

def main(args):
    if "dat.csv" not in listdir('.'):
        return
    if "images" not in listdir('.'):
        mkdir("images")
    append = ((args[-1] == "+") if len(args) > 0 else False)
    offset = 1
    if append:
        #Get current count of image files in folder
        offset += sum([1 for i in listdir('./images') if i[-(i[::-1].index(".")):] == "png"])

    with open("./dat.csv") as file:
        r = csv.reader(file)
        for row in list(r)[offset:]:
            #Obviously, this isn't practical for a large scale, but this is a quick and dirty way to get a small set of random images
            url = f"https://dummyimage.com/64.png/%s/%s/&text=%s" % (row[3], row[4], row[1])
            open(f"./images/%s.png" % row[0],"wb").write(get(url).content)

main(argv[1:])
