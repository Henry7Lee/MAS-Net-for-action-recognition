# MAS-Net for action recognition
Submitted in 2022 CVPR


#### Something-Something
Model  | Frames x Crop x Clip  |V1 Top-1 |V2 Top-1| 
:--: | :--: | :--: | :--:| 
MAS-ResNet50_8F  | 8x1x1      | 49.2%  | 61.5% | 
MAS-ResNet50_16F | 16x1x1     | 51.9%  | 63.0% | 
MAS-ResNet50_EN  | (8+16)x1x1 | 54.5%  | 65.1% | 
Model  | Frames x CropS x ClipS  |V1 Top-1 |V2 Top-1| 
:--: | :--: | :--: | :--:| 
MAS-ResNet50_8F  | 8x3x2      | 51.1%  | 63.9% | 
MAS-ResNet50_16F | 16x3x2     | 53.4%  | 65.1% | 
MAS-ResNet50_EN  | (8+16)x3x2 | 55.4%  | 66.7% | 

#### Kinetics-400
Model  | Frames x Crops x Clips   | Top-1 | Top-5  |
:--: | :--: | :--: | :--:| 
MAS-ResNet50    | 16x3x10 | 75.7%  | 92.6%  |
