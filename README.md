
# MAS-Net for action recognition

## Submitted in 2022 CVPR 

#### Something-Something:1-Crop & Center-Clip
Model| Frames x 1Crop x 1Clip |V1-Top1 |V2-Top1| 
:--: | :--: | :--: | :--:| 
MAS-ResNet50_8F  | 8 x 1 x 1      | 49.2%  | 61.5%  
MAS-ResNet50_16F | 16 x 1 x 1     | 51.9%  | 63.0%  
MAS-ResNet50_EN  | (8+16) x 1 x 1 | 54.5%  | 65.1%  

#### Something-Something:3-Crops & 2-Clips
Model| Frames x Crops x Clips |V1-Top1 |V2-Top1| 
:--: | :--: | :--: | :--:| 
MAS-ResNet50_8F  | 8 x 3 x 2      | 51.1%  | 63.9%  
MAS-ResNet50_16F | 16 x 3 x 2     | 53.4%  | 65.1%  
MAS-ResNet50_EN  | (8+16) x 3 x 2 | 55.4%  | 66.7%  

#### Kinetics-400
Model  | Frames x Crops x Clips   |&nbsp;   Top-1 &nbsp;  | &nbsp;  Top-5  &nbsp;  |
:--: | :--: | :--: | :--:| 
MAS-ResNet50_16F    | 16x3x10 | 75.7%  | 92.6%  
