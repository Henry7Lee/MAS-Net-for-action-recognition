# MAS-Net for action recognition
Submitted in 2022 CVPR


#### Something-Something
Model  | Frames x 1-Crop x 1-Clip |V1 Top-1 |V2 Top-1| 
:--: | :--: | :--: | :--:| 
MAS-ResNet50_8F  | 8 x 1 x 1      | 49.2%  | 61.5% | 
MAS-ResNet50_16F | 16 x 1 x 1     | 51.9%  | 63.0% | 
MAS-ResNet50_EN  | (8+16) x 1 x 1 | 54.5%  | 65.1% | 

Model  | Frames x Crop x Clip |V1 Top-1 |V2 Top-1| 
:--: | :--: | :--: | :--:| 
MAS-ResNet50_8F  | 8 x 3 x 2      | 51.1%  | 63.9% | 
MAS-ResNet50_16F | 16 x 3 x 2     | 53.4%  | 65.1% | 
MAS-ResNet50_EN  | (8+16) x 3 x 2 | 55.4%  | 66.7% | 

#### Kinetics-400
Model  | Frames x Crops x Clips   | Top-1 | Top-5  |
:--: | :--: | :--: | :--:| 
MAS-ResNet50    | 16x3x10 | 75.7%  | 92.6%  |
