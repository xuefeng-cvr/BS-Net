# Boundary-induced and scene-aggregated network for monocular depth prediction

Created by Junfeng Cao, Fei Sheng and Feng Xue.

### Introduction

Monocular depth prediction is an important task in scene understanding. It aims to predict the dense depth of a single RGB image.
Furthermore, it can be used for scene understanding and perception, such as object detection, segmentation, and 3D reconstruction.
Obtaining accurate depth information is a prerequisite for many computer vision tasks.

The Data Preparation and Evaluation are following Junjie Hu with [his work](https://github.com/JunjH/Revisiting_Single_Depth_Estimation).
Thanks for his valuable work.

### Citation

If you find BS-Net useful in your research, please consider citing:
```
@article{BSNet,
title = {Boundary-induced and scene-aggregated network for monocular depth prediction},
author = {Feng Xue and Junfeng Cao and Yu Zhou and Fei Sheng and Yankai Wang and Anlong Ming},
journal = {Pattern Recognition},
pages = {107901},
year = {2021}
}
```

## Dependencies
python 3.6  
Pytorch 1.0  
scipy 1.2.1  
h5py 3.0.0  
Pillow 6.0.0  
scikit-image 0.17.2  
scikit-learn 0.22.1

## Data Preparation

#### NYUD v2

You may download the dataset from [NYUD v2](https://drive.google.com/file/d/1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw/view?usp=sharing) and unzip it to the ./data folder. You will have the following directory structure:
```
BS-Net
|_ data
|   |-nyu2_train
|   |-nyu2_test
|   |-nyu2_train.csv
|   |-nyu2_test.csv
```

#### iBims-1

For iBims-1  dataset only have 100 RGB-D pictures especially designed for testing single-image depth estimation methods, you may download the dataset original images from [iBims-1](https://www.bgu.tum.de/lmf/ibims1/) . And you will have the following directory structure:
```
BS-Net
|_ data
|  |_ iBims1
|     |_ ibims1_core_raw
|     |_ ibims1_core_mat
|     |_ imagelist.txt
```
## Training

For training BS-Net on NYUD v2 training dataset, you can run:

```
python train.py
```
You can download our trained model from [Google Drive](https://drive.google.com/file/d/1r-xjxP-Ds5inGKRFHK9JX7VoplcFYiRH/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1WifbKL2_KQ11H1nsYrG3yw) (Code: 1jmz).

Note that: the performance of the given model is slightly different from the manuscript, which is represented as follows. And $\downarrow$ means smaller is better, and $\uparrow$ means larger is better.

**NYU D v2 dataset**

Depth accuracy and error:
|Meth|RMSE$\downarrow$|REL$\downarrow$|Lg10$\downarrow$|MAE$\downarrow$|Delta1$\uparrow$|Delta2$\uparrow$|Delta3$\uparrow$|
|---|---|---|---|---|---|---|---|
|this impl.|0.548|0.127|0.055|0.336|0.838|0.968|0.993|
|paper|0.550|0.123|0.053|-|0.846|0.969|0.992|

Depth boundary accuracy:
<table>
    <tr>
        <th rowspan="2">Meth</th>
        <th colspan="3">th<0.25</th>
        <th colspan="3">th<0.5</th>
        <th colspan="3">th<1</th>
    </tr>
    <tr>
        <td>P</td>
        <td>R</td>
        <td>F1</td>
        <td>P</td>
        <td>R</td>
        <td>F1</td>
        <td>P</td>
        <td>R</td>
        <td>F1</td>
    </tr>
    <tr>
        <td>this impl.</td>
        <td>0.642</td>
        <td>0.494</td>
        <td>0.552</td>
        <td>0.659</td>
        <td>0.496</td>
        <td>0.559</td>
        <td>0.737</td>
        <td>0.537</td>
        <td>0.613</td>
    </tr>
    <tr>
        <td>paper</td>
        <td>0.644</td>
        <td>0.483</td>
        <td>0.546</td>
        <td>0.665</td>
        <td>0.492</td>
        <td>0.558</td>
        <td>0.750</td>
        <td>0.531</td>
        <td>0.613</td>
    </tr>
</table>

Farthest region error under partition ratios:

|Meth|m=6|m=12|m=24|
|---|---|---|---|
|this impl.|0.110|0.132|0.141|
|paper|0.106|0.126|0.134|

**iBims-1 dataset**

Conventional depth error and accuracy:

|Meth|RMSE$\downarrow$|REL$\downarrow$|Lg10$\downarrow$|sq_rel$\downarrow$|Delta1$\uparrow$|Delta2$\uparrow$|Delta3$\uparrow$|
|---|---|---|---|---|---|---|---|
|this impl.|1.146|0.234|0.119|0.385|0.526|0.829|0.932|
|paper|1.190|0.240|0.120|-|0.51|0.82|0.93|

Planarity error, depth boundary errors, and directed depth error:

|Meth|pe_plan$\downarrow$|pe_ori$\downarrow$|deb_acc$\downarrow$|deb_com$\downarrow$|dde_0$\uparrow$|dde_m$\downarrow$|dde_p$\downarrow$|
|---|---|---|---|---|---|---|---|
|this impl.|4.426|31.06|2.059|5.263|81.17|16.92|1.909|
|paper|3.98|28.75|2.25|5.18|80.54|17.64|1.80|

Farthest region error under partition ratios:

|Meth|m=6|m=12|m=24|
|---|---|---|---|
|this impl.|0.173|0.189|0.218|
|paper|0.1724|0.1863|0.1981|

## Evaluation

For testing BS-Net on NYUD v2 testing dataset, you can run:
```
python test.py
```
or testing on iBims-1  dataset you can run:
```
python test_iBims1.py
```
