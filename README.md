# ResNet38-Watershed
ResNet38 to predict gradient direction of distance transform branch of watershed unified network.


## Implementation

Using (Model A1, 2convs) in paper [Wider or Deeper: Revisiting the ResNet Model for Visual Recognition](https://arxiv.org/abs/1611.10080) as base network.
Downsampling 3 times with convolution stride=2 before B2, B3 and B4. 
Use dilated convolution in B5(rate=2), B6(rate=4), B7(rate=4). 
Use dropout: dropout rate=0.3 for 2048 channels, dropout rate=0.5 for 4096 channels. 

Model structure: Input -> B0 -> B2(stride=2, x3) -> B3(stride=2, x3) -> B4(stride=2, x6) -> B5(rate=2, x3) -> B6(rate=4, x1) -> B7(rate=4, x1) -> Gating -> graddir/grad-convs1([3,3,4096,512], [3,3,512,512]) -> graddir/grad-convs2([1,1,512,256], [1,1,256,256], [1,1,256,2]) -> graddir/grad-norm

## Results

- Unified watershed gradient direction of distance transform.
- Result: 
- Data set: 2975 training image(1024x2048). 500 val images(not used for training). 1525 test images(without GT) 
- Data augmentation: Per image standardization (adapted from MXnet implementation). Randomly shuffle. Resize all images to [512,1024].
- Training: 
	- Stage1: Train 18 epochs. Batch 3. Adam optimizer(lr=0.00016). L2 weight decay 0.0005.
	- Stage2: Train 9 epoches. Batch 3. Adam optimizer(lr=0.00001). L2 weight decay 0.0005.
- Device: TitanX(Pascal) 12GB

## Acknowledge

Thanks for the GPU provided by [Computer Vision and Pattern Recongnition Group at Technical University Munich](https://vision.in.tum.de/)

## TODO

- Try Momentum Opt.

## NOTES

- MUST no random flip, otherwise the loss won't drop/converge.

