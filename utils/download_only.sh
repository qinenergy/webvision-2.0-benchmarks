#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: download.sh Download and Preprocess Webvision2018 
# Author: Qin Wang <wang@qin.ee>


for i in {0..9}
do
    wget https://data.vision.ee.ethz.ch/cvl/webvision2018/webvision_train_0$i.tar 
done
for i in {10..32}
do
    wget https://data.vision.ee.ethz.ch/cvl/webvision2018/webvision_train_$i.tar
done	

wget https://data.vision.ee.ethz.ch/cvl/webvision2018/val_images_resized.tar
wget https://data.vision.ee.ethz.ch/cvl/webvision2018/val_filelist.txt
wget https://data.vision.ee.ethz.ch/cvl/webvision2018/info.tar

wget https://data.vision.ee.ethz.ch/cvl/webvision2018/test_images_resized.tar                                                                                          
wget https://data.vision.ee.ethz.ch/cvl/webvision2018/test_filelist.txt

