import torch
from torch import nn

DROPOUT_RATE = xxxx 

#【0】 特征提取模块
class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding= 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(DROPOUT_RATE),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(DROPOUT_RATE)
        )
        
    def forward(self, x):
            return self.conv(x)

#【1】初始卷积模块
class in_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(in_conv,self).__init__()
        self.conv = double_conv(in_ch, mid_ch, out_ch)
        
    def forward(self, x):
            return self.conv(x)      

# 【2】输出卷积模块
class out_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, class_num):
        super(out_conv,self).__init__()
        self.conv = nn.Sequential(
            double_conv(in_ch, mid_ch, out_ch)
        )
        self.fc = nn.Conv2d(out_ch, class_num, 1, padding=0)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# 【3】上采样模块
class up_sampling(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor = 2,  mode = 'bilinear'):
        super(up_sampling,self).__init__()
        self.up = nn.Sequential( 
            nn.Upsample(scale_factor = scale_factor, mode = mode),
            nn.Conv2d(in_ch, out_ch, 1, padding=0))
        
    def forward(self, x):
            return self.up(x)

# 【4】Global feature 模块
class global_feature(nn.Module):
    def __init__(self, in_size):
        super(global_feature,self).__init__()
        self.gf = nn.Sequential( 
            nn.AdaptiveAvgPool2d(1),
            nn.Upsample(scale_factor = in_size))
        
    def forward(self, x):
            return self.gf(x)


from torch import nn
from torch.nn import functional as F

# 【5】搭建 RUnet 模型
class InvUnet(nn.Module):
    def __init__(self, in_ch, out_ch,in_size, class_num):
        super(InvUnet,self).__init__()
        self.input = in_conv(in_ch, 32,32)
        self.conv1 = nn.Sequential(nn.MaxPool2d(2), double_conv(32, 64, 64))
        self.conv2 = nn.Sequential(nn.MaxPool2d(2),double_conv(64, 128, 128))
        self.conv3 = nn.Sequential(nn.MaxPool2d(2),double_conv(128, 256, 256))
        self.conv4 = double_conv(32, 32, 32)
        self.conv5 = double_conv(32, 32, 32)
        self.conv6 = double_conv(32, 32, 32)
        self.output = out_conv(32, 32, 32,class_num)
        self.up1 = up_sampling(64, 32, scale_factor = 2)
        self.up2 = up_sampling(128, 32, scale_factor = 4)
        self.up3 = up_sampling(256, 32, scale_factor = 8)
        self.gf = global_feature(in_size)
        
         # 参数初始化
        for name,params in self.named_parameters():
            if name.find('weight') != -1 and len(params.size()) != 1:
                nn.init.xavier_uniform(params)
            if name.find('bias') != -1:
                nn.init.constant(params,0)

    def forward(self,x):
        x1 = self.input(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        
        gf = self.gf(x1)
        x1 = x1 + gf
        x1 = self.conv4(x1)
        x2 = self.up1(x2)
        x = x1 + x2
        x = self.conv5(x)
        x3 = self.up2(x3)
        x = x + x3
        x = self.conv6(x)
        x4 = self.up3(x4)
        x = x + x4
        x = self.output(x)
        return x
    

#【6】针对语义分割任务的 InvUnet
pxl_model = InvUnet(3, 32, 256, 2)

#【7】针对实例分割任务的 InvUnet-3c
ins_model = InvUnet(3, 32, 256, 3)
