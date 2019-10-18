from collections import OrderedDict
from models.base_model import BaseModel
from optimizers.radam import RAdam
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import sys

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        self.convtrans = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.bilinear = bilinear

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = nn.functional.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x1 = self.convtrans(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x



class UNet(nn.Module):
    """Standard U-Net architecture network.
    
    Input params:
        n_channels: Number of input channels (usually 1 for a grayscale image).
        n_classes: Number of output channels (2 for binary segmentation).
    """
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class Segmentation2DModel(BaseModel):
    def __init__(self, configuration):
        """Initialize the model.
        """
        super().__init__(configuration)

        self.loss_names = ['segmentation']
        self.network_names = ['unet']

        self.netunet = UNet(1, 2)
        self.netunet = self.netunet.to(self.device)
        if self.is_train:  # only defined during training time
            self.criterion_loss = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.netunet.parameters(), lr=configuration['lr'])
            self.optimizers = [self.optimizer]

        # storing predictions and labels for validation
        self.val_predictions = []
        self.val_labels = []
        self.val_images = []


    def forward(self):
        """Run forward pass.
        """
        self.output = self.netunet(self.input)


    def backward(self):
        """Calculate losses; called in every training iteration.
        """
        self.loss_segmentation = self.criterion_loss(self.output, self.label)


    def optimize_parameters(self):
        """Calculate gradients and update network weights.
        """
        self.loss_segmentation.backward() # calculate gradients
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()


    def test(self):
        super().test() # run the forward pass

        # save predictions and labels as flat tensors
        self.val_images.append(self.input)
        self.val_predictions.append(self.output)
        self.val_labels.append(self.label)


    def post_epoch_callback(self, epoch, visualizer):
        self.val_predictions = torch.cat(self.val_predictions, dim=0)
        predictions = torch.argmax(self.val_predictions, dim=1)
        predictions = torch.flatten(predictions).cpu()

        self.val_labels = torch.cat(self.val_labels, dim=0)
        labels = torch.flatten(self.val_labels).cpu()

        self.val_images = torch.squeeze(torch.cat(self.val_images, dim=0)).cpu()

        # Calculate and show accuracy
        val_accuracy = accuracy_score(labels, predictions)

        metrics = OrderedDict()
        metrics['accuracy'] = val_accuracy

        visualizer.plot_current_validation_metrics(epoch, metrics)
        print('Validation accuracy: {0:.3f}'.format(val_accuracy))

        # Here you may do something else with the validation data such as
        # displaying the validation images or calculating the ROC curve

        self.val_images = []
        self.val_predictions = []
        self.val_labels = []