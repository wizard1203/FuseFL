<!-- <p align="center"><img src="./figs/logo_with_name.png" width=90% height=60% ></p> -->

<h1 align="center">
Official code for the paper "FuseFL: One-Shot Federated Learning through the Lens of Causality with Progressive Model Fusion" (NeurIPS 2024 Spotlight).
</h1>

<a href="https://openreview.net/forum?id=E7fZOoiEKl"><img src="https://img.shields.io/badge/OpenReview-FuseFL-blue" alt="Paper"></a>
<a href="https://github.com/wizard1203/FuseFL"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>

## Introduction
In this work, we explore the reason why current one-shot FL methods have significant performance drop than FedAVG with many communication rounds. We find that easiliy fitting on the spurious correlations during local training is the main cause of this low performance. From the causal perspective, we observe that the spurious fitting can be alleviated by augmenting intermediate features from other clients. Then, we design FuseFL, which leverages progressive model fusion to implement feature augmentation while maintaining low communication costs as same as other one-shot FL methods, and achieve high performance. This repository contains the official PyTorch implementation of the FuseFL framework.

## How to Run the Basic Code
To run the basic code, execute the following command:
```bash
base scripts/cifar10/c5.sh
```
Note: Remember to modify the wandb name and entity as needed, and configure the data directory, Python path, etc.

## Change Hyper-parameters
The script `scripts/cifar10/c5.sh` provides examples for 5 clients, training with ensemble learning or FuseFL. You can refer to it to conveniently change hyper-parameters. More provided examples and other experiments can be found in `scripts/algs`.

## Installation
There is no other special library than PyTorch.

## Citation
If you find our work useful, please kindly cite our paper:

```
@inproceedings{tang2024fusefl,
    title={FuseFL: One-Shot Federated Learning through the Lens of Causality with Progressive Model Fusion},
    author={Zhenheng Tang and Yonggang Zhang and Peijie Dong and Yiu-ming Cheung and Amelie Chi Zhou and Bo Han and Xiaowen Chu},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=E7fZOoiEKl}
}
```




