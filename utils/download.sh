#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: download.sh Download and Preprocess Webvision2018 
# Author: Qin Wang <wang@qin.ee>


for i in {0..9}
do
    wget https://data.vision.ee.ethz.ch/aeirikur/webvision2018/webvision_train_0$i.tar 
    tar -xvf webvision_train_0$i.tar
    rm webvision_train_0$i.tar
done
for i in {10..32}
do
    wget https://data.vision.ee.ethz.ch/aeirikur/webvision2018/webvision_train_$i.tar
    tar -xvf webvision_train_$i.tar
    rm webvision_train_$i.tar	
done	

wget https://data.vision.ee.ethz.ch/cvl/webvision2018/val_images_resized.tar
tar -xvf val_images_resized.tar 
rm val_images_resized.tar 
wget https://data.vision.ee.ethz.ch/cvl/webvision2018/val_filelist.txt
wget https://data.vision.ee.ethz.ch/cvl/webvision2018/info.tar
tar -xvf info.tar
rm info.tar

# Process Train Set
mkdir train
mkdir meta
python3 preprocess.py

# Process Val Set
mv val_images_resized val
mv val_filelist.txt meta/val.txt


# Process Test Set
wget https://data.vision.ee.ethz.ch/cvl/webvision2018/test_images_resized.tar                            
tar -xvf test_images_resized.tar                                                                         
rm test_images_resized.tar                                                                   
wget https://data.vision.ee.ethz.ch/cvl/webvision2018/test_filelist.txt
mkdir webvision2018_test
mv test_images_resized webvision2018_test/val

mkdir webvision2018_test/meta

awk '$0=$0" 0"' test_filelist.txt > webvision2018_test/meta/val.txt
cp meta/synsets.txt webvision2018_test/meta/
