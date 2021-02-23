'''
Generate a given number of dummy images
Generate corresponding csv file with number of loops

Call:
python generator.py [n] (test?: t for true) (append: + for True) (display?: d for true)
'''
from random import triangular, randint
from requests import get
from string import printable
from sys import argv
import csv
import os


#subs = {"+":"%2B","#":"%23","%":"%25","&","%26"}
chars = list(printable)[:95]
chars[64:68] = ["%23","$","%25","%26"]
chars[72] = "%2B"
chars = chars[:91] + chars[92:]#Get rid of |
loops = [int(i) for i in list("1000101021110110100000001110000000001201000000000011110000000000122200000000000000010000000000")]

def main(args):
    n = int(args[0])
    disp = (args[-1] == "d" or
        (len(args) > 2 and args[-2] == "d") or
        (len(args) > 3 and args[-3] == "d"))
    test = (args[-1] == "t" or
        (len(args) > 2 and args[-2] == "t") or
        (len(args) > 3 and args[-3] == "t"))
    target = "./" + ("test_" if test else "") + "dat.csv"
    append = (args[-1] == "+" and os.path.exists(target))
    offset = 0
    #if we're appending, we want the indices to start at the appropriate row
    if append:
        offset = len(list(csv.reader(open(target)))) - 1

    output = []
    print(f"Generating %d entries" % n)
    print(f"Appending: %r" % append)
    print(f"Target CSV: %s\n" % target)

    print("Generating...")
    if disp:
        print("Index\tLoops\tText")
    for i in range(offset, n + offset):
        #Generate string of text and corresponding count of loops
        length = int(triangular(1,11,6))
        nums = [randint(0,93) for i in range(length)]
        text = "".join([chars[i] for i in nums])
        while "/." in text:
            text = text.replace("/.", "./")
        count = sum([loops[i] for i in nums])
        if disp:
            print("%d\t%d\t%s" % (i, loops, text))
        #Generate foreground and background with 1 or 0 color channels similar
        b_fore = [(randint(0,1)==1) for i in range(3)]
        b_back = [not(i) for i in b_fore]
        if (index := randint(0,3)) != 3:
            b_back[index] = not(b_back[index])
        fore = "".join(["0f"[int(i)] for i in b_fore])
        back = "".join(["0f"[int(i)] for i in b_back])

        output.append([i,text,count,fore,back])
    print("Generation finished\n")
    #we use the with function here, since it will close the writer as soon as we're done with it
    print("Writing to %s" % target)
    with open(target, ("a" if append else "w"), newline = '') as csvfile:
        writer = csv.writer(csvfile)
        if not(append):
            writer.writerow(["file_index","text","loops","foreground","background"])
        for line in output:
            writer.writerow(line)
    print("Writing finished")

main(argv[1:])
