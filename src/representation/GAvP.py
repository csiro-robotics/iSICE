import torch
import torch.nn as nn

class GAvP(nn.Module):
     """Global Average pooling
        Widely used in ResNet, Inception, DenseNet, etc.
     """
     def __init__(self, input_dim=2048, dimension_reduction=None):
         super(GAvP, self).__init__()
         self.dr = dimension_reduction
         if self.dr is not None:
             self.conv_dr_block = nn.Sequential(
               nn.Conv2d(input_dim, self.dr, kernel_size=1, stride=1, bias=False),
               nn.BatchNorm2d(self.dr),
               nn.ReLU(inplace=True)
             )
         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
         self.output_dim = self.dr if self.dr else input_dim
         self._init_weight()

     def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

     def forward(self, x):
         if self.dr is not None:
             x = self.conv_dr_block(x)
         x = self.avgpool(x)
         return x
