# aWSFE

## Dependencies

- Python 3.6 (Anaconda3 Recommended)
- Pytorch 1.0.1
- torchvision 0.2.1
- CUDA 10.1

## Datasets

### CIFAR-100

It will be downloaded automatically by torchvision when running the code.

### ImageNet-Sub

Download the ILSVRC2012 from the [Link](http://image-net.org). Then following [UCIR](https://github.com/hshustc/CVPR19_Incremental_Learning), create the ImageNet-Sub dataset.

## Getting Started

For CIFAR-100, N=5, |exemplar|=2K: 

```python
python aWSFE_cifar100.py  --nb_cl_fg 50 --nb_cl 10 --nb_protos 20  --resume  --random_seed 1993 --T 2  --ckp_prefix seed_1993_class_incremental_aWSFE_cifar100  --num_workers 4 --epochs 160
```

For ImageNet-100, N=5, |exemplar|=2K: 

```python
python aWSFE_imagenet.py  --nb_cl_fg 50 --nb_cl 10 --nb_protos 20  --resume  --random_seed 1993 --T 2  --ckp_prefix seed_1993_class_incremental_aWSFE_imagenet  --num_workers 4 --epochs 90 --datadir <your_imagenet_sub_path>
```



For other evaluation scenarios (e.g. N=2 or T=10), please modify nb_cl or other corresponding setting.

## Acknowledgements

- [Learning a Unified Classifier Incrementally via Rebalancing](https://github.com/hshustc/CVPR19_Incremental_Learning)

- [SS-IL : Separated Softmax for Incremental Learning](https://github.com/hongjoon0805/SS-IL-Official)

