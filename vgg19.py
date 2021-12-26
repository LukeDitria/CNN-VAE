import torch.nn as nn


class VGG19(nn.Module):
    """
     Simplified version of the VGG19 "feature" block
     This module's only job is to return the "feature loss" for the inputs
    """
    def __init__(self, channel_in, width=64):
        super(VGG19, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, width, 3, 1, 1)
        self.conv2 = nn.Conv2d(width, width, 3, 1, 1)
        
        self.conv3 = nn.Conv2d(width, 2 * width, 3, 1, 1)
        self.conv4 = nn.Conv2d(2 * width, 2 * width, 3, 1, 1)
        
        self.conv5 = nn.Conv2d(2 * width, 4 * width, 3, 1, 1)
        self.conv6 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv7 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv8 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        
#         self.conv9 = nn.Conv2d(4 * width, 8 * width, 3, 1, 1)
#         self.conv10 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
#         self.conv11 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
#         self.conv12 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        
        # Add/Remove layers of feature extractor - ToDo: need a better way of doing this now
        # With larger images you may want to use more layers
        
#         self.conv13 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
#         self.conv14 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
#         self.conv15 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
#         self.conv16 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.relu = nn.ReLU(inplace=True)
        
    def feature_loss(self, x):
        diff = x[:x.shape[0]//2] - x[x.shape[0]//2:]
        return diff.pow(2).mean()
        
    def forward(self, x):
        """
        :param x: Expects x the target and source to concatenated on dimension 0
        :return: Average feature loss
        """
        x1 = self.relu(self.conv1(x))
        loss = self.feature_loss(x1)
        x2 = self.relu(self.conv2(x1))
        loss += self.feature_loss(x2)
        x2 = self.mp(x2)

        x3 = self.relu(self.conv3(x2))
        loss += self.feature_loss(x3)
        x4 = self.relu(self.conv4(x3))
        loss += self.feature_loss(x4)
        x4 = self.mp(x4)

        x5 = self.relu(self.conv5(x4))
        loss += self.feature_loss(x5)
        x6 = self.relu(self.conv6(x5))
        loss += self.feature_loss(x6)
        x7 = self.relu(self.conv7(x6))
        loss += self.feature_loss(x7)
        x8 = self.relu(self.conv8(x7))
        loss += self.feature_loss(x8)
#         x8 = self.mp(x8)

#         x9 = self.relu(self.conv9(x8))
#         loss += self.feature_loss(x9)
#         x10 = self.relu(self.conv10(x9))
#         loss += self.feature_loss(x10)
#         x11 = self.relu(self.conv11(x10))
#         loss += self.feature_loss(x11)
#         x12 = self.relu(self.conv12(x11))
#         loss += self.feature_loss(x12)
#         x12 = self.mp(x12)

#         x13 = self.relu(self.conv13(x12))
#         loss += self.feature_loss(x13)
#         x14 = self.relu(self.conv14(x13))
#         loss += self.feature_loss(x14)
#         x15 = self.relu(self.conv15(x14))
#         loss += self.feature_loss(x15)
#         x16 = self.relu(self.conv16(x15))
#         loss += self.feature_loss(x16)

        return loss/8
        