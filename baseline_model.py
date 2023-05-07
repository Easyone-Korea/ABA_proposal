import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np
# torch.nn.init.xavier_normal_()
import torch.nn.init as init
class EEGNET_baseline(nn.Module):
    def __init__(self, num_classes, input_ch, input_time, batch_norm=True,
                 batch_norm_alpha=0.1):
        super().__init__()
        # print("input_ch", input_ch)
        # print("input_time", input_time)
        # print("num_classes", num_classes)
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = num_classes
        freq = 1000
        self.convnet_com = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, freq // 2), stride=1, bias=False, padding=(0, freq // 4)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=(input_ch, 1), stride=1, groups=8),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.25),
            nn.Conv2d(16, 16, kernel_size=(1, freq // 4), padding=(0, freq // 8), groups=16),
            nn.Conv2d(16, 16, kernel_size=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.25)
        )
        for m in self.convnet_com.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)


        out = self.convnet_com(torch.zeros(1, 1, input_ch, input_time))
        self.num_hidden = out.size()[1] * out.size()[2] * out.size()[3]

        self.fc = nn.Sequential(
            nn.Linear(self.num_hidden, num_classes),
            nn.Dropout()
        )
        init.xavier_uniform_(self.fc[0].weight)

    def forward(self, x):
        # print("x0", x.shape)
        output = self.convnet_com(x)
        # print("output1", output.shape)
        output = output.view(output.size()[0], -1)
        # print("output2", output.shape)
        output = self.fc(output)
        # print("output3", output.shape)
        output = output.view(output.size()[0], self.n_classes, -1)
        # print("output4", output.shape)
        if output.size()[2] == 1:
            output = output.squeeze(2)
            # print("output5", output.shape)
        return output