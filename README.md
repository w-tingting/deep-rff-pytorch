# deep-rff-pytorch

This repository implements SDCRKL-GP (Scalable Deep Convolutional Random Kernel Learning in Gaussian Process for Image Recognition).

# Installation
python 3.7.4
pytorch 1.7.0

# Run
```
git clone https://github.com/w-tingting/deep-rff-pytorch.git
cd experiments
python run.py
```

# Evaluate
```
cd experiments
python test.py
```
# Result
|     datasets      | error rate (%) | nlpp  | Parameters | FLOPs   | MAC     |
| :---------------: | :------------: | ----- | ---------- | ------- | ------- |
|       MNIST       |      0.60      | 0.020 | 19.088k    | 0.984M  | 1.057M  |
|      FMNIST       |      7.22      | 0.229 | 12.176k    | 0.336M  | 0.892M  |
|      CIFAR10      |     27.28      | 0.811 | 22.448k    | 2.454M  | 1.582M  |
|     CALTECH4      |      2.81      | 0.086 | 47.024k    | 15.434M | 19.021M |
|     CIFAR100      |     60.68      | 2.468 | -          | -       | -       |
| tiny-ImageNet-200 |     73.52      | 3.746 | -          | -       | -       |

