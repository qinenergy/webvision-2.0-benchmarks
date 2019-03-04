# Benchmark Models for WebVision 2.0 Dataset


## Step-1: Prepare data
The data directory should be prepared following the structure stated in this [link](https://tensorpack.readthedocs.io/modules/dataflow.dataset.html#tensorpack.dataflow.dataset.ILSVRC12). 
We provided a download + preprocess script for this.
```
cd utils
bash download.sh
```

## Step-2: Train the model (ResNet-50)
```
python3 ./imagenet-resnet.py --data /raid/webvision2018/ --gpu 0,1,2,3 -d 50 --mode resnet
```

## Pretrained models
We offer two pretrained ResNet-50 models. Due to the class imbalance in WebVision, we duplicated the file items in train.txt such that different classes have equal number of training samples.

[520000 Steps (101 ImageNet Epoch)](https://drive.google.com/open?id=12359rElqF1GBLp8AhDPtcV6pdPw9jkbx)   30.69% Top5 Balanced Class Error Rate - Validation Set

[1055000 Steps (205 ImageNet Epoch)](https://drive.google.com/open?id=1Rsf0TFgbC6CmPyQfaBchil_guJxj1MIl)   28.51% Top5 Balanced Class Error Rate - Validation Set

The 205 Epoch model achieves Top5 28.37% error rate on the development set in CodaLab. 

## Evaluation and Submission
To generate the prediction files for CodaLab submissions, assume testimages are stored in the above format in /raid/webvision2018_test/val/:
```
python3 ./imagenet-resnet.py --data /raid/webvision2018_test/ -d 50 --mode resnet --eval --load train_log/imagenet-resnet-d50-webvision2018-200epochs/model-1055000

# Prepariing the submission file
python3 utils/json2sub.py  
```

## Dependencies
+ Tensorflow
+ Tensorpack
+ opencv-python

## Acknowledgement
ImageNet ResNet-50 Code from Tensorpack, we modified the data pipeline for Webvision.



