# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 16:14:37 2021

@author: Administrator
"""

import torch
import torch.nn as nn
from ecb import ECB


class Main(nn.Module):
	def __init__(self,channel=32):
		super(Main,self).__init__()

		self.conv_in = nn.Conv2d(3,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_out = nn.Conv2d(channel,3,kernel_size=3,stride=1,padding=1,bias=False)

		self.repb1 = Block(channel)
		self.repb2 = Block(channel)
		self.repb3 = Block(channel)
		self.repb4 = Block(channel)
		self.repb5 = Block(channel)

		self.krm = KRM(channel)
        

	def forward(self,x):
        
		x_in = self.conv_in(x)
           
		x_1 = self.repb1(self.krm(x_in) + x_in)
		x_2 = self.repb2(self.krm(x_1) + x_1)        
		x_3 = self.repb3(self.krm(x_2) + x_2)
		x_4 = self.repb4(self.krm(x_3) + x_3)
		x_5 = self.repb5(self.krm(x_4) + x_4)
        
		x_out = self.conv_out(x_5)
        
		return x_out


class Block(nn.Module):
    def __init__(self,channel):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(channel,channel,kernel_size=3,padding=1,bias=True)
        self.act = nn.PReLU(channel)
        self.conv2= nn.Conv2d(channel,channel,kernel_size=3,padding=1,bias=True)
        self.conv3= nn.Conv2d(channel,channel,kernel_size=3,padding=1,bias=True)
        self.conv4= nn.Conv2d(channel,channel,kernel_size=3,padding=1,bias=True)
        self.cbam = CBAMLayer(channel)
        
        self.norm = nn.GroupNorm(num_channels=channel,num_groups=1)


    def forward(self, x):
             
        res1 = self.act(self.norm(self.conv1(x)))
        res2 = self.act(self.norm(self.conv2(res1)))  
        cbam = self.cbam(res2)
        res3 = self.act(self.norm(self.conv3(cbam)))          
        res4 = self.act(self.norm(self.conv4(res3)) + x)
        
        return res4
    
    
class KRM(nn.Module):    
	def __init__(self,channel):                                
		super(KRM,self).__init__()

		self.conv_in = nn.Conv2d(channel,channel//4,kernel_size=1,stride=1,padding=0,bias=False)
		self.ecbb_t1 = ECB(channel//4, channel//4, depth_multiplier=2.0)
		self.conv_out = nn.Conv2d(channel//4,channel,kernel_size=1,stride=1,padding=0,bias=False)
        
	def forward(self,x):
        
		x_t = self.conv_out(self.ecbb_t1(self.conv_in(x)))
		
		return	x_t
    

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=8, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(

            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x

        return x
