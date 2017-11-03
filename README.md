# ResNet38-Watershed
ResNet38 to perform semantic branch of watershed unified network.


## Implementation

Implement original Model A1, 2convs in paper [Wider or Deeper: Revisiting the ResNet Model for Visual Recognition](https://arxiv.org/abs/1611.10080).
Downsampling 3 times with convolution stride=2 before B2, B3 and B4.
Use dilated convolution in B5(rate=2), B6(rate=4), B7(rate=4) and the last two convolution layers(rate=12)
Use dropout: dropout rate=0.3 for 2048 channels, dropout rate=0.5 for 4096 channels.

Model A structure: Input -> B0 -> B2(stride=2, x3) -> B3(stride=2, x3) -> B4(stride=2, x6) -> B5(rate=2, x3) -> B6(rate=4, x1) -> B7(rate=4, x1) ->
Semantic(BN+2convs, rate=12) -> Softmax

## Results

- Unified watershed semantic branch which implements semantic segmentation.
- Pretrain weights: from imgnet.
- Best accuracy on semantic: 66.76%
- Data set: 2975 training image(1024x2048). 500 val images(not used for training). 1525 test images(without GT) 
- Data augmentation: 
	- Per image standardization (adapted from MXnet implementation). 
	- Randomly resize with a ratio [0.7, 1.3], preserve original height-weight ratio. 
	- Randomly crop to [504,504].
	- Randomly shuffle.
	- Batch 4
- Training: 
	- Stage1: Train 150 epochs. Batch 4. Momentum optimizer(lr=0.0016, momentum=0.9, fixed). L2 weight decay 0.0005. Init from imgnet.
	  - Acc: 66.76%. (66.23% for 140 epoches)
	- Stage2: Train 64 epochs. Batch 4. Momentum optimizer(lr=0.0008, momentum=0.9, linear). L2 weight decay 0.0005. Init from stage1. 
	  - Acc: ??
- Device: Quadro P6000 24GB

## Acknowledge

Thanks for the GPU provided by [Computer Vision and Pattern Recongnition Group at Technical University Munich](https://vision.in.tum.de/)

## TODO

- Stage1 loss could be further reduced.

## NOTES

- BN params: 
	- Do not update moving statistics of shared conv layers, but update gamma/betas.
	- Do update moving statistics and gamma/betas of semantic unique parts.

- Batch Size:
	- Batch 4 is much better than batch 3, though batch 3 might achieve a lower loss.
