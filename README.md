## [Non-Binary IoU and Progressive Coupling and Reﬁning Networkfor Salient Object Detection]
by Qianwei Zhou, Chen Zhou, Zihao Yang, Yingkun Xu, Qiu Guan

# Introduction
Recently, many salient object detection (SOD) methods decouple image features into body features and edge features, whichimply a new development direction in the ﬁeld of SOD. Most of them mainly focus on how to decouple features, but the fusionmethod for the decoupled features can be further improved. In this paper, we propose a network, namely Progressive Coupling andReﬁning Network (PCRNet), which allows the progressive coupling and reﬁning of the decoupled features to get accurate salientfeatures. Furthermore, a novel loss, namely Non-Binary Intersection over Union (NBIoU), is proposed based on the characteristicsof non-binary label images and the principle of Intersection over Union (IoU) loss. Experimental results show that our NBIoUperformance surpasses binary cross-entropy (BCE), IoU and Dice on non-binary label images. The results on ﬁve popular SODbenchmark datasets show that our PCRNet signiﬁcantly exceeds the previous state-of-the-art (SOTA) methods on multiple metrics.In addition, although our method is designed for SOD, it is comparable with previous SOTA methods on multiple benchmarkdatasets for camouﬂaged object detection without any modiﬁcation on the network structure, veriﬁed the robustness of the proposedmethod. The code will be released upon acceptance.
![PCF](./fig/PCF.png)
## Prerequisites
- [Python 3.5](https://www.python.org/)
- [Pytorch 1.3](http://pytorch.org/)
- [OpenCV 4.0](https://opencv.org/)
- [Numpy 1.15](https://numpy.org/)
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [Apex](https://github.com/NVIDIA/apex)


## Clone repository
```shell
git clone https://github.com/QianWeiZhou/PCRNet.git
cd PCRNet/
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
- Train the model and get the predicted masps and salient maps, which will be saved into `data/DUTS`
```shell
    cd script/
    python3 train.py
    python3 test.py
```
- Use above predicted maps to train the model again and predict final saliency maps, which will be saved into `data/testset` folder.
```shell
    cd /script/
    python3 train.py
    python3 test.py
```


## Testing 
- If you just want to evaluate the performance of LDF without training, please download our trained model into `train-fine/out` folder
- Predict the saliency maps
```shell
    python3 test.py
```

## Saliency maps & Trained model
- saliency maps: #谷歌/百度网盘

## Citation
- If you find this work is helpful, please cite our paper
```
#bibtex引用
```
