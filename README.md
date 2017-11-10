# ResNet38-Watershed
ResNet38 to perform discretized watershed transform branch of watershed unified network.
Note that this is a rather small network.

## Implementation

Implement based on Model A1, 2convs in paper [Wider or Deeper: Revisiting the ResNet Model for Visual Recognition](https://arxiv.org/abs/1611.10080).
Downsampling 2 times with convolution stride=2 at B2, B3. No dilated convs.

Model structure: Input(grad-gt, sem-gt, full size) - > Gate -> B0[3,3,2,8] -> B2[[3,3,8,16],[3,3,16,16], downsample] -> B3[[3,3,16,32],[3,3,32,32], downsample]
-> B4[[3,3,32,64],[3,3,64,64]] -> Tail[[3,3,64,64],[3,3,64,16]] -> Softmax
## Results

- Unified watershed wt branch which implements discretized watershed transform.
- Init: Randomly init
- Performance: Visually good
- Data set: 2975 training image(1024x2048). 500 val images(not used for training). 1525 test images(without GT) 
- Data augmentation: Per image standardization (adapted from MXnet implementation). Randomly shuffle.
- Training: Train 18 epochs. Batch 6. Adam optimizer(rate=0.0005). L2 weight decay 0.0005.
- Device: TitanX(Pascal) 12GB

## Acknowledge

Thanks for the GPU provided by [Computer Vision and Pattern Recongnition Group at Technical University Munich](https://vision.in.tum.de/)

## NOTES
