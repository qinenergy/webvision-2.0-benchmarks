# WebVision-baseline


## Preparing data
Data directory should have the structure stated in this [link](https://tensorpack.readthedocs.io/modules/dataflow.dataset.html#tensorpack.dataflow.dataset.ILSVRC12). 
We provide a download + Preprocess script download.sh in utils.
```
bash download.sh
```

## Train Model from Scratch
```
python3 ./imagenet-resnet.py --data /raid/webvision2018/ --gpu 0,1,2,3 -d 50 --mode resnet
```

## Pretrained model
We offer two pretrained ResNet-50 models. Both using upsampling in train.txt to balance training between classes.

[520000 Steps (101 ImageNet Epoch)](https://drive.google.com/open?id=12359rElqF1GBLp8AhDPtcV6pdPw9jkbx)   30.69% Top5 Balanced Class Error Rate

[1055000 Steps (205 ImageNet Epoch)](https://drive.google.com/open?id=1Rsf0TFgbC6CmPyQfaBchil_guJxj1MIl)   28.51% Top5 Balanced Class Error Rate

The 205 Epoch model achieves Top5 28.37% error rate on the development set in CodaLab. 

## Evaluation and Submission
To generate the prediction files for CodaLab submissions, assume testimages are stored in the above format in /raid/webvision2018_test/val/:
```
python3 ./imagenet-resnet.py --data /raid/webvision2018_test/ -d 50 --mode resnet --eval --load train_log/imagenet-resnet-d50-webvision2018-200epochs/model-1055000

# Prepariing the submission test file
python3 utils/json2sub.py  
```

## Main Libraries
Tensorflow
Tensorpack
opencv-python

## Acknowledgement
ImageNet ResNet-50 Code from Tensorpack, we modified the data pipeline for Webvision.



