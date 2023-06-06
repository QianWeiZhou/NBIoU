## [Non-Binary IoU and Progressive Coupling and Reﬁning Networkfor Salient Object Detection]
by Qianwei Zhou, Chen Zhou, Zihao Yang, Yingkun Xu, Qiu Guan

# Introduction
Recently, many salient object detection (SOD) methods decouple image features into body features and edge features, whichimply a new development direction in the ﬁeld of SOD. Most of them mainly focus on how to decouple features, but the fusionmethod for the decoupled features can be further improved. In this paper, we propose a network, namely Progressive Coupling andReﬁning Network (PCRNet), which allows the progressive coupling and reﬁning of the decoupled features to get accurate salientfeatures. Furthermore, a novel loss, namely Non-Binary Intersection over Union (NBIoU), is proposed based on the characteristicsof non-binary label images and the principle of Intersection over Union (IoU) loss. Experimental results show that our NBIoUperformance surpasses binary cross-entropy (BCE), IoU and Dice on non-binary label images. The results on ﬁve popular SODbenchmark datasets show that our PCRNet signiﬁcantly exceeds the previous state-of-the-art (SOTA) methods on multiple metrics.In addition, although our method is designed for SOD, it is comparable with previous SOTA methods on multiple benchmarkdatasets for camouﬂaged object detection without any modiﬁcation on the network structure, veriﬁed the robustness of the proposedmethod. The code will be released upon acceptance.

## Prerequisites
- install miniconda
- $ conda env create -f environment.yaml

## Clone repository
```shell
git clone https://github.com/QianWeiZhou/NBIoU.git
```

## Download dataset
Download the following datasets and unzip them into `data` folder

- [PASCAL-S](http://cbi.gatech.edu/salobj/)
- [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
- [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html)
- [DUT-OMRON](http://saliencydetection.net/dut-omron/)
- [DUTS](http://saliencydetection.net/duts/)

## Training & Evaluation
- If you want to train the model by yourself, please download the [pretrained model](https://download.pytorch.org/models/resnet50-19c8e357.pth) into `res` folder
- If you want to train the model with RGB dataset
- Train the model and get the predicted masps and salient maps, which will be saved into `data/DUTS`
```shell
    cd PCRet_RGB/
    python3 train.py
```
- If you want to train the model with COD dataset
```shell
    cd PCRNet_COD/
    python3 train.py
```

## Testing 
- Predict the saliency maps
```shell
    python3 test.py
```

## Saliency maps & Trained model
- saliency maps: https://pan.baidu.com/s/1Pyu7JPEga5FndpPXNXn2ug?pwd=fgxn code：fgxn 


## Citation
- If you find this work is helpful, please cite our paper
```
@article{ZHOU2023120370,
title = {Non-binary IoU and progressive coupling and refining network for salient object detection},
journal = {Expert Systems with Applications},
volume = {230},
pages = {120370},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.120370},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423008722},
author = {Qianwei Zhou and Chen Zhou and Zihao Yang and Yingkun Xu and Qiu Guan},
keywords = {Salient object detection, Intersection over Union, Label decouple},
abstract = {Recently, many salient object detection (SOD) methods decouple image features into body features and edge features, which imply a new development direction in the field of SOD. Most of them mainly focus on how to decouple features, but the fusion method for the decoupled features can be further improved. In this paper, we propose a network, namely Progressive Coupling and Refining Network (PCRNet), which allows the progressive coupling and refining of the decoupled features to get accurate salient features. Furthermore, a novel loss, namely Non-Binary Intersection over Union (NBIoU), is proposed based on the characteristics of non-binary label images and the principle of Intersection over Union (IoU) loss. Experimental results show that our NBIoU performance surpasses binary cross-entropy (BCE), IoU and Dice on non-binary label images. The results on five popular SOD benchmark datasets show that our PCRNet significantly exceeds the previous state-of-the-art (SOTA) methods on multiple metrics. In addition, although our method is designed for SOD, it is comparable with previous SOTA methods on multiple benchmark datasets for camouflaged object detection without any modification on the network structure, verified the robustness of the proposed method.}
}
```
