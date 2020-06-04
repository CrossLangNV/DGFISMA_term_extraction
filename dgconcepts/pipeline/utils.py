def beautiprint(title,list, toFile=None):
    """ Print out a list with a title """
    print(title, file=toFile)
    print("#######################", file=toFile)
    print("", file=toFile)
    for l in list:        
        print(l, file=toFile)
    print("", file=toFile)   