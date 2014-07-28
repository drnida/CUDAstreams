import sys
from xml.dom import minidom
f = open ('Comments.xml')
g = open ('parsedComments.xml', 'a')
for line in f:
    #print line

    try:
        xmldoc = minidom.parseString(line)
        
        #text = xmldoc.getElementsByTagName('row')[0].getAttribute('Text')
        #print text
        g.write(xmldoc.getElementsByTagName('row')[0].getAttribute('Text'))
    except KeyboardInterrupt:
        quit()
    except:
        pass


g.close()
f.close()
"""
for x in text:
    print s.attributes['Text'].value
"""
