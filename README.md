# ResNet38-Watershed
ResNet38 to perform semantic and instance segmentation on Cityscape dataset simultaneously.


## Implementation

Implement original Model A1, 2convs in paper [Wider or Deeper: Revisiting the ResNet Model for Visual Recognition](https://arxiv.org/abs/1611.10080).
Downsampling 3 times with convolution stride=2 before B2, B3 and B4.
Use dilated convolution in B5(rate=2), B6(rate=4), B7(rate=4) and the last two convolution layers(rate=12)
Use dropout: dropout rate=0.3 for 2048 channels, dropout rate=0.5 for 4096 channels.

Model A structure: Input -> B0 -> B2(stride=2, x3) -> B3(stride=2, x3) -> B4(stride=2, x6) -> B5(rate=2, x3) -> B6(rate=4, x1) -> B7(rate=4, x1) ->
Tail(BN+2convs, rate=12) -> Softmax

## Results

- watershed branch implements unified watershed network.
- Best accuracy on semantic: ??
- Best accuracy on instance: ??
- Data set: 2975 training image(1024x2048). 500 val images(not used for training). 1525 test images(without GT) 
- Data augmentation: Per image standardization (adapted from MXnet implementation)  randomly crop? Per epoch randomly shuffle?
- Training: Train ?? epochs. Batch ??. Adam optimizer(rate=0.001 -> ). L2 weight decay 0.0005.
- Device: Quadro P6000 24GB

## Acknowledge

Thanks for the GPU provided by [Computer Vision and Pattern Recongnition Group at Technical University Munich](https://vision.in.tum.de/)

## TODO

- 
