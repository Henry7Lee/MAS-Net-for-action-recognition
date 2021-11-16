import time
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet50', 'resnet101']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class MSM(nn.Module):
    def __init__(self, input_channel, num_segments=8,attention_mode="Channel"):
        super(MSM, self).__init__()
        self.num_segments=num_segments

        if attention_mode =="Channel":
            self.kernel_size = (1,1,1)
            self.padding = (0,0,0)
        elif attention_mode == "Spatial":
            self.kernel_size = (1,3,3)
            self.padding = (0,1,1)
        elif attention_mode == "Temporal":
            self.kernel_size = (3,1,1)
            self.padding = (1,0,0)

        self.reduction = 4
        self.groups =  16//self.reduction

        self.squeeze_channel = input_channel // self.reduction

        self.conv_fc1 = nn.Conv3d(input_channel, self.squeeze_channel, kernel_size=self.kernel_size,
                                  stride=(1, 1, 1), padding=self.padding, bias=False,groups=self.groups)

        self.bn_fc1   = nn.BatchNorm3d(self.squeeze_channel)
        self.conv_fc2 = nn.Conv3d(self.squeeze_channel, 2 *input_channel, kernel_size=self.kernel_size,
                                  stride=(1, 1, 1), padding=self.padding, bias=False, groups=self.groups)

        self.relu = nn.ReLU()


    def forward(self,d1,d2):
        nt, c, h, w = d1.size()
        n_batch = nt // self.num_segments
        d1 = d1.view(n_batch, self.num_segments, c, h, w).transpose(2,1).contiguous() #[n,c,t,h,w]
        d2 = d2.view(n_batch, self.num_segments, c, h, w).transpose(2,1).contiguous()

        d1_pool=nn.functional.avg_pool3d(d1, kernel_size=[self.num_segments, 1, 1]) #[n,c,1,h,w]
        d2_pool=nn.functional.avg_pool3d(d2, kernel_size=[self.num_segments, 1, 1])

        d = d1_pool+d2_pool  #[n,c,1,h,w]
        d = self.conv_fc1(d)  #[n,c//r,1,h,w]
        if n_batch !=1:
            d = self.bn_fc1(d)
            d = self.relu(d)
        d = self.conv_fc2(d) #[n,2c,1,h,w]

        d = torch.unsqueeze(d, 1).view(-1, 2, c, 1, h, w) #[n,2,c,1,h,w]
        d = nn.functional.softmax(d, 1)
        d1 = d1 * d[:, 0, :, :, :, :].squeeze(1)
        d2 = d2 * d[:, 1, :, :, :, :].squeeze(1)
        d  = d1 + d2
        d = d.transpose(2,1).contiguous().view(nt, c, h, w)
        return d



class MAM(nn.Module):

    def __init__(self, input_channel, num_segments=8):
        super(MAM, self).__init__()

        self.input_channel = input_channel
        self.reduction = 64
        self.squeeze_channel = self.input_channel // self.reduction
        self.num_segments = num_segments
        self.conv1 = nn.Conv2d(in_channels=self.input_channel,out_channels=self.squeeze_channel,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.squeeze_channel)
        #self.tanh = nn.Tanh()

        self.conv3 = nn.Conv2d(in_channels=self.squeeze_channel,out_channels=self.input_channel, kernel_size=1,bias=False)
        #self.bn3 = nn.BatchNorm2d(num_features=self.input_channel)

        self.alpha=0.5


    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.num_segments

        bottleneck = self.conv1(x)  # [nt,c//r,h,w]
        bottleneck = self.bn1(bottleneck)  # [nt,c//r,h,w]

        reshape_bottleneck_avg = bottleneck.view((-1, self.num_segments) + bottleneck.size()[1:])  # [n,t,c//r,h,w]

        mapping0, last_frame = reshape_bottleneck_avg.split([self.num_segments - 1, 1], dim=1)  # [n,t-1,c//r,h,w] # [n,1,c//r,h,w]
        __, mapping1 = reshape_bottleneck_avg.split([1, self.num_segments - 1], dim=1)  # [n,t-1,c//r,h,w]

        avg_foward = (mapping0 + self.alpha*mapping1)/(1.0 + self.alpha) # [n,t-1,c//r,h,w]
        avg_foward_pad =torch.cat((avg_foward, last_frame),dim=1) #[n,t,c//r,h,w]

        avg_foward_conv = avg_foward_pad.view((-1,) + avg_foward_pad.size()[2:])  # [nt,c//r,h,w]
        y = self.conv3(avg_foward_conv)  # [nt,c,h,w]

        return y

class TIM(nn.Module):
    def __init__(self, input_channels, n_segment=8, stride=1,n_div=8, mode='shift'):
        super(TIM, self).__init__()
        self.input_channels = input_channels
        self.stride = stride
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div
        self.conv = nn.Conv1d(self.input_channels, self.input_channels,
                kernel_size=3, padding=1, groups=self.input_channels,
                bias=False)

        if mode == 'shift':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1 # shift left
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right
            if 2*self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1 # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1 # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x = x.permute(0, 3, 4, 2, 1) # [n,h,w,c,t]
        x = x.contiguous().view(n_batch*h*w, c, self.n_segment) # [n*h*w,c,t]
        x = self.conv(x) # [n*h*w,c,t]

        x = x.view(n_batch, h, w, c, self.n_segment)
        x = x.permute(0, 4, 3, 1, 2) # [n,t,c,h,w]
        x = x.contiguous().view(nt, c, h, w) # [nt,c,h,w]
        return x



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.num_segments = num_segments


        self.MAM = MAM(planes, self.num_segments)

        self.MSM = MSM(planes, self.num_segments,attention_mode="Spatial") #Temporal #Channel #Spatial

        self.TIM = TIM(planes, n_segment=self.num_segments, stride=stride, n_div=8, mode='shift')

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # [nt,c,h,w]

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        Motion = self.MAM(out)

        out = self.MSM(Motion,out)
        out = self.TIM(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, num_segments,block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.num_segments = num_segments

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.num_segments,block, 64, layers[0])
        self.layer2 = self._make_layer(self.num_segments,block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(self.num_segments,block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(self.num_segments,block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, num_segments ,block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(num_segments, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(num_segments, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #Grad-CAM
        #self.layerout = x.detach().cpu()

        adaptiveAvgPoolWidth = x.shape[2]
        x = nn.functional.avg_pool2d(x, kernel_size=adaptiveAvgPoolWidth)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet50(num_segments=8,**kwargs):
    """Constructs a ResNet-50 based model.
    """
    model = ResNet(num_segments, Bottleneck, [3, 4, 6, 3],**kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet50'])#, model_dir='./pretrained/')
    model.load_state_dict(checkpoint, strict=False)
    return model

def resnet101(num_segments):
    """Constructs a ResNet-101 model.
    Args:
        groups
    """
    model = ResNet(num_segments,Bottleneck, [3, 4, 23, 3])
    checkpoint = model_zoo.load_url(model_urls['resnet101'])#, model_dir='./pretrained/')
    model.load_state_dict(checkpoint, strict=False)
    return model




if __name__ == '__main__':
    model = resnet50(num_segments=8)
    Input = torch.randn([8, 3, 224, 224])  # N,C,T,W,H
    time1=time.time()
    out = model(Input)
    print(time.time()-time1)
    from thop import profile, clever_format
    flops, params = profile(model, inputs=(Input,))
    flops, params = clever_format([flops, params], '%.3f')
    print("FLOPs {}, params {}".format(flops, params))

