import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import fvcore.nn.weight_init as weight_init


class GAM(nn.Module):
    def __init__(self, channel_in=512):

        super(GAM, self).__init__()
        self.query_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 
        self.key_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 

        self.scale = 1.0 / (channel_in ** 0.5)

        self.conv6 = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 

        for layer in [self.query_transform, self.key_transform, self.conv6]:
            weight_init.c2_msra_fill(layer)

    def forward(self, x5):
        # x: B,C,H,W
        # x_query: B,C,HW
        B, C, H5, W5 = x5.size()

        x_query = self.query_transform(x5).view(B, C, -1)

        # x_query: B,HW,C
        x_query = torch.transpose(x_query, 1, 2).contiguous().view(-1, C) # BHW, C
        # x_key: B,C,HW
        x_key = self.key_transform(x5).view(B, C, -1)

        x_key = torch.transpose(x_key, 0, 1).contiguous().view(C, -1) # C, BHW

        # W = Q^T K: B,HW,HW
        x_w = torch.matmul(x_query, x_key) #* self.scale # BHW, BHW
        x_w = x_w.view(B*H5*W5, B, H5*W5)
        x_w = torch.max(x_w, -1).values # BHW, B
        x_w = x_w.mean(-1)
        #x_w = torch.mean(x_w, -1).values # BHW
        x_w = x_w.view(B, -1) * self.scale # B, HW
        x_w = nn.functional.softmax(x_w, dim=-1) # B, HW
        x_w = x_w.view(B, H5, W5).unsqueeze(1) # B, 1, H, W
 
        x5 = x5 * x_w
        x5 = self.conv6(x5)

        return x5


class SelfAtt(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAtt, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class SpatialGroupEnhance(nn.Module):

    def __init__(self, groups=8):
        super().__init__()
        self.groups=groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight=nn.Parameter(torch.zeros(1,groups,1,1))
        self.bias=nn.Parameter(torch.zeros(1,groups,1,1))
        self.sig=nn.Sigmoid()
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h,w=x.shape
        x=x.view(b*self.groups,-1,h,w) #bs*g,dim//g,h,w
        xn=x*self.avg_pool(x) #bs*g,dim//g,h,w
        xn=xn.sum(dim=1,keepdim=True) #bs*g,1,h,w
        t=xn.view(b*self.groups,-1) #bs*g,h*w

        t=t-t.mean(dim=1,keepdim=True) #bs*g,h*w
        std=t.std(dim=1,keepdim=True)+1e-5
        t=t/std #bs*g,h*w
        t=t.view(b,self.groups,h,w) #bs,g,h*w
        
        t=t*self.weight+self.bias #bs,g,h*w
        t=t.view(b*self.groups,1,h,w) #bs*g,1,h*w
        x=x*self.sig(t)
        x=x.view(b,c,h,w)

        return x 


class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
