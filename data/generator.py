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
from pandas import DataFrame, read_csv, concat
import os


#subs = {"+":"%2B","#":"%23","%":"%25","&","%26"}
chars = list(printable)[:95]
chars[64:68] = ["%23","$","%25","%26"]
chars[72] = "%2B"
chars = chars[:91] + chars[92:]#Get rid of |
loops = [int(i) for i in list("1000101021110110100000001110000000001201000000000011110000000000122200000000000000010000000000")]

def main(args):
    n = 100
    d = False
    t = False
    append = False
    for arg in args:#Takes in command line args
        try: arg = int(arg)
        except: pass
        if type(arg) == int:
            n = arg
        elif arg == "+":
            append = True
        elif arg == "d":
            d = True
        elif arg == "t":
            t = True
        elif arg == "+":
            append = True
        else:
            print(f"Argument '%s' ignored" % str(arg))
    target = "./" + ("test_" if t else "") + "dat.csv"
    append = (append and os.path.exists(target))
    offset = 0
    #if we're appending, we want the indices to start at the appropriate row
    if append:
        df = read_csv(target)
        offset = len(df)

    output = []
    print(f"Generating %d entries" % n)
    print(f"Appending: %r" % append)
    print(f"Target CSV: %s\n" % target)

    print("Generating...")
    if d:
        print("Index\tLoops\tText")

    for i in range(offset, n + offset):
        #Generate string of text and corresponding count of loops
        length = int(triangular(1,11,6))
        nums = [randint(0,93) for i in range(length)]
        text = "".join([chars[i] for i in nums])
        while "/." in text:
            text = text.replace("/.", "./")
        count = sum([loops[i] for i in nums])
        if d:
            print("%d\t%d\t%s" % (i, count, text))

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
    if not 'df' in locals():
        df = DataFrame(columns = ["file_index","text","loops","foreground","background"])
    # with open(target, "w", newline = '') as csvfile:
    df_output = DataFrame(output, columns = list(df.columns))
    df = concat([df, df_output])
    df.to_csv(target, index = False)
    print("Writing finished")

main(argv[1:])
