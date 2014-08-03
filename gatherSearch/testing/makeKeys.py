
#Makes keys from a file.
#Doesn't strip utf-8 elipses and some  
#non printing symbols
#
#



#f = open("filetest.txt",'r')
#f = open("TwitSample.txt",'r')
f = open("parsedComments.txt",'r')
#'a' arg appends to file 
g = open("key_list.txt", 'a')

line = f.read()
tokens = line.split()

keys = set()

for x in tokens:
    # how to strip utf-8 elipse ... no one knows...
    keys.add( x.strip("\n'.?!*()[]:,/;{}").strip('"') )


#print keys
for x in keys:
    #print x 
    g.write("1234" + x + "\n") 


f.close()
g.close()
