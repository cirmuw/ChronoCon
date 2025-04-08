import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn



#from torchvision.models.vgg import model_urls   # CW: https://github.com/clovaai/CRAFT-pytorch/issues/191 # had to be removed



VGG_TYPES = {'vgg11' : torchvision.models.vgg11,
             'vgg11_bn' : torchvision.models.vgg11_bn,
             'vgg13' : torchvision.models.vgg13,
             'vgg13_bn' : torchvision.models.vgg13_bn,
             'vgg16' : torchvision.models.vgg16,
             'vgg16_bn' : torchvision.models.vgg16_bn,
             'vgg19_bn' : torchvision.models.vgg19_bn,
             'vgg19' : torchvision.models.vgg19}


class Custom_VGG(nn.Module):

    def __init__(self,
                 ipt_size=(128, 128),
                 pretrained=True,
                 vgg_type='vgg19_bn',
                 num_classes=1000,
                 regression=False,
                 ordinal=False):
        super(Custom_VGG, self).__init__()

        # load convolutional part of vgg
        assert vgg_type in VGG_TYPES, "Unknown vgg_type '{}'".format(vgg_type)
        vgg_loader = VGG_TYPES[vgg_type]
        vgg = vgg_loader(pretrained=pretrained)
        self.features = vgg.features

        # init fully connected part of vgg
        test_ipt = Variable(torch.zeros(1,3,ipt_size[0],ipt_size[1]))
        test_out = vgg.features(test_ipt)
        self.n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self.classifier = None
        if regression == True:
            self.classifier = nn.Sequential(nn.Linear(self.n_features, 4096),
                                            nn.ReLU(True),
                                            nn.Dropout(),
                                            nn.Linear(4096, 4096),
                                            nn.ReLU(True),
                                            nn.Dropout(),
                                            nn.Linear(4096, 1)
                                           )
        elif ordinal == True:
            self.classifier = nn.Sequential(nn.Linear(self.n_features, 4096),
                                            nn.ReLU(True),
                                            nn.Dropout(),
                                            nn.Linear(4096, 4096),
                                            nn.ReLU(True),
                                            nn.Dropout(),
                                            nn.Linear(4096, num_classes),
                                            nn.Sigmoid()
                                           )
        else:
            self.classifier = nn.Sequential(nn.Linear(self.n_features, 4096),
                                            nn.ReLU(True),
                                            nn.Dropout(),
                                            nn.Linear(4096, 4096),
                                            nn.ReLU(True),
                                            nn.Dropout(),
                                            nn.Linear(4096, num_classes),
                                            nn.Softmax(dim=1)
                                           )

        self._init_classifier_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _init_classifier_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
