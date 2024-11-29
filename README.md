# Leveraging Eye Movement for Instructing Robust Video-based Facial Expression Recognition

This repository is the Pytorch implementation for  **Leveraging Eye Movement for Instructing Robust Video-based Facial Expression Recognition**.


## 0. Contents

1. Requirements
2. Data Preparation
3. Pre-trained Models
4. Training
5. Evaluation

## 1. Requirements

To install requirements:
Python Version: 3.7.9

## 2. Data Preparation

You need to download the related datasets  and put in the folder which namely dataset.

[DFEW](https://dfew-dataset.github.io/)

[MAFW](https://github.com/MAFW-database/MAFW)

[CASMEⅡ、CASMEⅢ](http://casme.psych.ac.cn/casme/)


## 3. Pre-trained Models

You can download our trained models from [Baidu Drive](xxx).

## 4. Evaluation


### 4.1 Macro

To evaluate on DFEW, run:

```
python macro.py --dataset DFEW
```

### 4.2 Micro

To evaluate on CASMEⅢ, run:

```
python micro.py --dataset CASME3
```
