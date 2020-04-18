#!/bin/sh
wget https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/summer2winter_yosemite.zip
unzip summer2winter_yosemite.zip
rm -rf summer2winter_yosemite.zip
mv summer2winter_yosemite data
cd data
mv testA/* trainA/
rm -rf testA
mv testB/* trainB/
rm -rf testB
mv trainA set1
mv trainB set2
cd set1
ls -v | cat -n | while read n f; do mv -n "$f" "$n.jpg"; done 
cd ../set2
ls -v | cat -n | while read n f; do mv -n "$f" "$n.jpg"; done
cd ../..
mkdir decodedArray
mkdir encodedArray
