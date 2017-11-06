# ResNet38-Watershed
ResNet38 to perform discretized watershed transform branch of watershed unified network.


## Implementation

Implement based on Model A1, 2convs in paper [Wider or Deeper: Revisiting the ResNet Model for Visual Recognition](https://arxiv.org/abs/1611.10080).
Downsampling 3 times with convolution stride=2 before B2, B3 and B4.
Use dilated convolution in B5(rate=2), B6(rate=4), B7(rate=4) and wt-convs
Use dropout: dropout rate=0.3 for 2048 channels, dropout rate=0.5 for 4096 channels.

Model A structure: Input -> B0 -> B2(stride=2, x3) -> B3(stride=2, x3) -> B4(stride=2, x6) -> B5(rate=2, x3) -> B6(rate=4, x1) -> B7(rate=4, x1) -> Gating -> Grad-convs(grad-convs1, grad-convs2, no dilation) -> WT-B0([3,3,2,64]) -> WT-B2([3,3,64,128], [3,3,128,128], dilate=12) -> WT-B3([3,3,128,256], [3,3,256,256], dilate=12) -> WT-B4([3,3,256,512], [3,3,512,512], dilated=12) -> WT-tail([3,3,512,512], [3,3,512,16], dilated=12) -> Softmax

## Results

- Unified watershed wt branch which implements discretized watershed transform.
- Pretrained: init from watershed-grads-27 epoches
- Performance: Not good
- Data set: 2975 training image(1024x2048). 500 val images(not used for training). 1525 test images(without GT) 
- Data augmentation: Per image standardization (adapted from MXnet implementation). Randomly shuffle. Per image resize to [512,1024]
- Training: Train 18 epochs. Batch 1. Adam optimizer(rate=5e-5). L2 weight decay 0.0005.
- Device: TitanX(Pascal) 12GB

## Acknowledge

Thanks for the GPU provided by [Computer Vision and Pattern Recongnition Group at Technical University Munich](https://vision.in.tum.de/)

## NOTES

- Train all BN parameters of graddir/wt branches, fix moving statistics of shared features.
- No randomly flip image while training.

