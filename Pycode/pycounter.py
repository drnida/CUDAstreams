
#Prints key value pairs for tokens in a file.
#
#



#f = open("filetest.txt",'r')
#f = open("TwitSample.txt",'r')
f = open("UnicodeSample.txt",'r')



line = f.read()
tokens = line.split()

keys = set()

for x in tokens:
    # how to strip utf-8 elipse ... no one knows...
    keys.add( x.strip("\n'.?!*()[]:,/;{}").strip('"') )


words = dict.fromkeys(keys, 0)


#f = open("filetest.txt",'r')
#f = open("UnicodeSample.txt",'r')
f.seek(0,0)
line = f.read()
tokens = line.split()

for x in tokens:
    y = x.strip("\n'.?!*()[]:,/;{}").strip('"') 
    words[y] = words[y] + 1


for x in words:
    print str(x) + " | " + str(words[x])


f.close()
