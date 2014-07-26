


f = open('CommentsKeyValue.txt','r')


threshhold = 1000

for line in f:
    values = line.split("|")
    #if int(values[1]) >= threshhold:
    try:
        if int(values[1]) >= threshhold:
            print line
    except:
        pass
        #print "TROUBLE " + line + "TROUBLE"
