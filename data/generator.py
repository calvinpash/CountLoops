'''
Generate a given number of dummy images
Generate corresponding csv file with number of loops

Call:
python generator.py [n] (append: + for True)
'''
from random import triangular, randint
from requests import get
from string import printable
from os import listdir
from sys import argv
import csv


#subs = {"+":"%2B","#":"%23","%":"%25","&","%26"}
chars = list(printable)[:95]
chars[64:68] = ["%23","$","%25","%26"]
chars[72] = "%2B"
loops = [int(i) for i in list("10001010211101101000000011100000000012010000000000111100000000001222000000000000000010000000000")]

def main(args):
    n = int(args[0])
    append = (args[-1] == "+" and "./dat.csv" in listdir('.'))
    offset = 0
    #if we're appending, we want the indices to start at the appropriate row
    if append:
        offset = len(list(csv.reader(open("./dat.csv")))) - 1

    output = []
    for i in range(n):
        #Generate string of text and corresponding count of loops
        length = int(triangular(1,11,6))
        nums = [randint(0,94) for i in range(length)]
        text = "".join([chars[i] for i in nums])
        count = sum([loops[i] for i in nums])

        #Generate foreground and background with 1 or 0 color channels similar
        b_fore = [(randint(0,1)==1) for i in range(3)]
        b_back = [not(i) for i in b_fore]
        if (index := randint(0,3)) != 3:
            b_back[index] = not(b_back[index])
        fore = "".join(["0f"[int(i)] for i in b_fore])
        back = "".join(["0f"[int(i)] for i in b_back])

        output.append([i + offset,text,count,fore,back])

    #we use the with function here, since it will close the writer as soon as we're done with it
    with open("./dat.csv", ("a" if append else "w"), newline = '') as csvfile:
        writer = csv.writer(csvfile)
        if not(append):
            writer.writerow(["file_index","text","loops","foreground","background"])
        for line in output:
            writer.writerow(line)

main(argv[1:])
