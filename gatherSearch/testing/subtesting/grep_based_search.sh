#!/bin/bash
while read line
do
   blah=$(grep -o "$line" parsedComments.txt | wc -l)
   echo "$blah $line"
done
