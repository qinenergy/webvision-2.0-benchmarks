#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: preprocess.py
# Author: Qin Wang <wang@qin.ee>

import os

# Construct Synsets
synsets = []
with open("./info/synsets.txt", "r") as f:
    rows = f.readlines()

with open("./meta/synsets.txt", "w") as f:
    for row in rows:
        k = row.split()[0]
        f.write(k + "\n")
        synsets.append(k)

# Construct Training Folders
try:
    for k in synsets:
        os.mkdir("./train/"+k)
except FileExistsError:
    pass


# Moving Files from Original Folders to train folders
filelist = []
with open("./info/train_filelist_all.txt", "r") as f:
    rows = f.readlines()

length = len(rows)
for i, row in enumerate(rows):
    if i%(length//200) == 0:
        print("Finished:", 100.*i/length, "%")
    source, label = row.strip().split()
    shortname = source.split("/")[-1]
    target = "./train/" + synsets[int(label)] + "/" + shortname
    try:
        os.rename(source, target)
        filelist.append(target[8:] + " " + label + "\n")
    except:
        print("Warning: FileNotExists", source)


# Construct Training Filelist
with open("./meta/train.txt", "w") as f:
    for row in filelist:
        f.write(row)




    
