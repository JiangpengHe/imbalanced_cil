# Gradient Reweighting: Towards Imbalanced Class-Incremental Learning [CVPR 2024] 

This repository is the official PyTorch implementation of **Gradient Reweighting: Towards Imbalanced Class-Incremental Learning**([arxiv]https://arxiv.org/abs/2402.18528).

## Training Instructions

To run the code, execute the provided bash script with the following format:

```
bash main.sh <dataset> <approach> <gpu> <scenario> <initial_classes> <total_tasks>
```

### Parameter Specifications:

- `<dataset>`: Dataset to be used (Options: `cifar100`, `imagenet_subset`, `imagenet_1000`, `Food101`).
- `<approach>`: Approach to be used, located in `./src/approaches/`.
  - `DGR`: our proposed Decoupled Gradients Reweighting (DGR) method.
- `<gpu>`: Index of the GPU to run the experiment.
- `<scenario>`: Learning scenario, including:
  - `conv`: Conventional CIL.
  - `lt`: Shuffled long-tailed scenario (experiments in main paper).
  - `ltio`: Ordered long-tailed scenario.
- `<initial_classes>`: Number of classes in the first base task.
- `<total_tasks>`: Total number of tasks (including the base task).

### Examples:

- Shuffled long-tailed case with LFS, 20 tasks on CIFAR100:
```
bash main.sh cifar100 DGR 0 lt 5 20
```

- Shuffled long-tailed case with LFH, 10 tasks on ImageNetSubset:
```
bash main.sh imagenet_subset DGR 0 lt 50 11
```

- Ordered long-tailed case with LFH, 5 tasks on Food101:
```
bash main.sh Food101 DGR 0 ltio 50 6
```

- Conventional case with LFS, 10 tasks on ImageNet_1000:
```
bash main.sh imagenet_1000 DGR 0 conv 10 10
```

### Acknolwedgement:
The main implementation of this repository is borrowed from "Long-Tailed Class-Incremental Learning": https://github.com/xialeiliu/Long-Tailed-CIL
