'''
Descripttion: 改写tensorflow中的GIDC_Unet模型为pytorch框架模型
version: 1.0
Author: luxin
Date: 2024-03-10 22:34:35
LastEditTime: 2024-03-12 11:16:27
'''
import torch
import torch.nn as nn

class GIDC_Unet(nn.Module):
    def __init__(self):
        super(GIDC_Unet, self).__init__()

        # Define the encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),  # conv0
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2),  # conv1
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),  # conv2
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  # conv3
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),  # conv4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),  # conv5
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Define the decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),  # conv6
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128 * 2, 64, kernel_size=5, stride=2, padding=2, output_padding=1),  # conv7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64 * 2, 32, kernel_size=5, stride=2, padding=2, output_padding=1),  # conv8
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32 * 2, 16, kernel_size=5, stride=2, padding=2, output_padding=1),  # conv9
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(16 * 2, 1, kernel_size=5, stride=1, padding=2),  # conv10
            nn.Sigmoid()
        )

    def forward(self, inpt):
        # Pass input through encoder
        enc_output = self.encoder(inpt)
        
        # Pass encoder output through decoder
        dec_output = self.decoder(enc_output)

        return dec_output
