# Gradient Reweighting: Towards Imbalanced Class-Incremental Learning [CVPR 2024] 

This repository is the official PyTorch implementation of **Gradient Reweighting: Towards Imbalanced Class-Incremental Learning**([arxiv]https://arxiv.org/abs/2402.18528).

## Training Instructions

To run the code, execute the provided bash script with the following format:

```
bash main.sh <dataset> <approach> <gpu> <scenario> <initial_classes> <total_tasks> <memory_budget> [<results_dir>]
```

### Parameter Specifications:

- `<dataset>`: Dataset to be used (Options: `cifar100`, `imagenet_subset`, `imagenet_1000`, `Food101`).
- `<approach>`: Approach to be used, located in `./src/approaches/`.
- `<gpu>`: Index of the GPU to run the experiment.
- `<scenario>`: Learning scenario, including:
  - `conv`: Conventional CIL.
  - `lt`: Shuffled long-tailed scenario (experiments in main paper).
  - `ltio`: Ordered long-tailed scenario.
- `<initial_classes>`: Number of classes in the first base task.
- `<total_tasks>`: Total number of tasks (including the base task).
- `<memory_budget>`: Number of exemplars per class.
- `<results_dir>`: (Optional) Directory for results. Default: `./results`.

### Examples:

- Shuffled long-tailed case with LFS, 20 tasks, 20 exemplars/class on CIFAR100:
```
bash main.sh cifar100 gradient_reweighting 0 lt 5 20 20
```

- Shuffled long-tailed case with LFH, 10 tasks, 20 exemplars/class on ImageNetSubset:
```
bash main.sh imagenet_subset gradient_reweighting 0 lt 50 11 20
```

- Ordered long-tailed case with LFH, 5 tasks, 20 exemplars/class on Food101:
```
bash main.sh Food101 gradient_reweighting 0 ltio 50 6 20
```

- Conventional case with LFS, 10 tasks, 50 exemplars/class on ImageNet_1000:
```
bash main.sh imagenet_1000 gradient_reweighting 0 conv 10 10 50
```
```
