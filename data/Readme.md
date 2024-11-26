## Dataset Preparation

This directory is designated for storing the datasets required for running the experiments. It's important to note that the datasets themselves are not included in this repository and should be prepared as outlined below.

## Dataset Folder Structure

For each dataset used in the experiments (except cifar100, which will be downloaded automatically), please include two files: `train.txt` and `test.txt`. 

Each line in both `train.txt` and `test.txt` files should follow the format:

```
[filepath] [space] [class_label]
```

For example, an entry in the `Food101` dataset might look like this:

```
/data/Food101/1.jpg 1
```
Here, `/data/Food101/1.jpg` is the path to the image file, and `1` is the corresponding class label.

## Cifar100
The dataset will be downloaded automatically to `./cifar100/`

## ImageNetSubset
Include the training file: train.txt and the test file: test.txt in `./ImageNetSubset/`

## ImageNet 
Include the training file: train.txt and the test file: test.txt in `./imagenet_1000/` 

## Food-101
Include the training file: train.txt and the test file: test.txt in `./Food101/`

