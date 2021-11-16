import torch.nn as nn
from torch.nn.init import normal_, constant_

from ops.basic_ops import ConsensusModule
from ops.transforms import *

class TemporalModel(nn.Module):
    def __init__(self, num_class, num_segments, base_model="MASNet",
                 backbone='resnet50',new_length=None,
                 consensus_type='avg',before_softmax=True,
                 dropout = 0.8,img_feature_dim=256,
                 full_res=False, partial_bn= True,
                 fc_lr5=False,print_spec=True,
                 ):

        super(TemporalModel, self).__init__()
        self.num_segments = num_segments #分段的数量
        self.base_model = base_model
        self.backbone = backbone
        self.dropout = dropout
        self.before_softmax = before_softmax
        self.reshape = True
        self.full_res = full_res

        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame

        self.fc_lr5 = fc_lr5  # fine_tuning for UCF/HMDB
        self.target_transforms = {86:87,87:86,93:94,94:93,166:167,167:166}


        if not self.before_softmax and self.consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1
        else:
            self.new_length = new_length

        if print_spec:
            print(("""Initializing TSN with base model: {}.
                      TSN Configurations:
                      input_modality:     {}
                      num_segments:       {}
                      new_length:         {}
                      consensus_module:   {}
                      dropout_ratio:      {}
                      img_feature_dim:    {}""".format(base_model, "RGB",
                                                       self.num_segments, self.new_length,
                                                       self.consensus_type, self.dropout,
                                                       self.img_feature_dim)))


        self._prepare_base_model(backbone)

        feature_dim = self._prepare_tsn(num_class)

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()


        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)


    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model,
                              self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Dropout(p=self.dropout))
            if self.consensus_type in ['TRN', 'TRNmultiscale']:
                # create a new linear layer as the frame feature
                self.new_fc = nn.Linear(feature_dim, self.img_feature_dim)
            else:
                # the default consensus types in TSN
                self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(
                getattr(self.base_model,
                        self.base_model.last_layer_name).weight, 0, std)
            constant_(
                getattr(self.base_model, self.base_model.last_layer_name).bias,
                0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim



    def _prepare_base_model(self, backbone):
        print(('=> backbone: {}'.format(backbone)))
        if 'resnet' in backbone:

            import ops.MASNet
            self.base_model = getattr(ops.MASNet, backbone)(self.num_segments)

            self.base_model.last_layer_name = 'fc'  # 将模型最后一层的名称改为'fc'
            self.input_size = 224
            if self.full_res:
                self.input_size = 256
            self.init_crop_size = 256
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

        else:
            raise ValueError('Unknown model: {}'.format(backbone))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TemporalModel, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable


    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []
        inorm = []
        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))

            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        if self.fc_lr5: # fine_tuning for UCF/HMDB
            return [
                {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
                'name': "first_conv_weight"},
                {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
                'name': "first_conv_bias"},
                {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
                'name': "normal_weight"},
                {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
                'name': "normal_bias"},
                {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
                'name': "BN scale/shift"},
                {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
                'name': "custom_ops"},
                {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
                'name': "lr5_weight"},
                {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
                'name': "lr10_bias"},
            ]
        else : # default
            return [
                {'params': first_conv_weight, 'lr_mult':  1, 'decay_mult': 1,
                'name': "first_conv_weight"},
                {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
                'name': "first_conv_bias"},
                {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
                'name': "normal_weight"},
                {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
                'name': "normal_bias"},
                {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
                'name': "BN scale/shift"},
                {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
                'name': "custom_ops"},
            ]



    def forward(self, input, no_reshape=False):
        #input.shape [4,24,224,224]
        if not no_reshape:
            sample_len = 3 * self.new_length  #[32,3,224,224]
            base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))  #[4*8,2048]
        else:
            base_out = self.base_model(input)


        if self.dropout > 0:
            base_out = self.new_fc(base_out)  #[32,174]

        if not self.before_softmax:
            base_out = self.softmax(base_out)

        if self.reshape:
              #[8, 174]
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            #base_out = base_out.view((-1, self.num_segments * num_crop) + base_out.size()[1:])
            output = self.consensus(base_out)
            return output.squeeze(1)



    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * self.init_crop_size // self.input_size

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
        else:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                            GroupRandomHorizontalFlip_sth(self.target_transforms)])





if __name__ == '__main__':
    model = TemporalModel(174, num_segments=8)