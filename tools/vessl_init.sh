#!/bin/bash

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &
mkdir data
cd data
mkdir coco
cd coco
wget http://images.cocodataset.org/zips/val2014.zip &
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip &
cd ..
cd ..
git clone -b vessl https://github.com/bscho333/Decoding.git
