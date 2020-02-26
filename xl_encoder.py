#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as utils
import math, copy
import time
from torch.autograd import Variable
from transformer_model_conv import *
from nnlib_conv_xl import *

class Compressor(nn.Module):
    def __init__(self,dim_input,d_model):
        super(Compressor,self).__init__()
        self.conv1d = nn.Conv1d(in_channels=dim_input,out_channels=d_model,kernel_size=32,stride=16,padding=0)

    def forward(self,x): 
        #x needs be (Batch,dim_in,SeqL)
        #print(self.conv1d.weight.data.size())
        y = self.conv1d(x)
        return y

class Expander(nn.Module):
    def __init__(self,d_model,d_hidden):
        super(Expander,self).__init__()
        self.convT1d = nn.ConvTranspose1d(in_channels=d_model,out_channels=d_hidden,kernel_size=32,stride=16,padding=0)

    def forward(self,x): 
        #x needs be (batch,dim_in,SeqL)
        y = self.convT1d(x)
        return y

class multiCompressor(nn.Module):
    def __init__(self,dim_input,d_model):
        super(multiCompressor,self).__init__()
        #TDNN-like Conv
        #self.conv1d1 = nn.Conv1d(in_channels=dim_input,out_channels=d_model,kernel_size=5,stride=1,padding=0)
        #self.conv1d2 = nn.Conv1d(in_channels=d_model,out_channels=d_model,kernel_size=4,stride=1,padding=0)
        #self.conv1d3 = nn.Conv1d(in_channels=d_model,out_channels=d_model,kernel_size=7,stride=1,padding=0)
        #self.conv1d4 = nn.Conv1d(in_channels=d_model,out_channels=d_model,kernel_size=10,stride=1,padding=0)
        #Strategy0
        #self.conv1d1 = nn.Conv1d(in_channels=dim_input,out_channels=d_model,kernel_size=4,stride=4,padding=0)
        #self.conv1d2 = nn.Conv1d(in_channels=d_model,out_channels=d_model,kernel_size=4,stride=4,padding=0)
        #self.conv1d3 = nn.Conv1d(in_channels=d_model,out_channels=d_model,kernel_size=4,stride=4,padding=0)
        #Strategy1
        self.conv1d1 = nn.Conv1d(in_channels=dim_input,out_channels=d_model,kernel_size=5,stride=3,padding=0)
        self.conv1d2 = nn.Conv1d(in_channels=d_model,out_channels=d_model,kernel_size=4,stride=2,padding=0)
        self.conv1d3 = nn.Conv1d(in_channels=d_model,out_channels=d_model,kernel_size=4,stride=4,padding=0)

    def forward(self,x): 
        #x needs be (Batch,dim_in,SeqL)
        #print(self.conv1d.weight.data.size())
        x1 = self.conv1d1(x)
        x2 = self.conv1d2(x1)
        #x3 = self.conv1d3(x2)
        #y = self.conv1d4(x3)
        y = self.conv1d3(x2)
        return y

class multiExpander(nn.Module):
    def __init__(self,d_model,d_hidden):
        super(multiExpander,self).__init__()
        #Strategy0
        #self.convT1d1 = nn.ConvTranspose1d(in_channels=d_model,out_channels=d_hidden,kernel_size=4,stride=4,padding=0)
        #self.convT1d2 = nn.ConvTranspose1d(in_channels=d_hidden,out_channels=d_hidden,kernel_size=4,stride=4,padding=0)
        #self.convT1d3 = nn.ConvTranspose1d(in_channels=d_hidden,out_channels=d_hidden,kernel_size=4,stride=4,padding=0)
        #Strategy1
        self.convT1d1 = nn.ConvTranspose1d(in_channels=d_model,out_channels=d_hidden,kernel_size=4,stride=4,padding=0)
        self.convT1d2 = nn.ConvTranspose1d(in_channels=d_hidden,out_channels=d_hidden,kernel_size=4,stride=2,padding=0)
        self.convT1d3 = nn.ConvTranspose1d(in_channels=d_hidden,out_channels=d_hidden,kernel_size=5,stride=3,padding=0)

    def forward(self,x): 
        #x needs be (batch,dim_in,SeqL)
        x1 = self.convT1d1(x)
        x2 = self.convT1d2(x1)
        y = self.convT1d3(x2)
        return y

class Generator(nn.Module):
    #Define standard linear+softmax generation step
    def __init__(self,d_model,d_out):
        super(Generator,self).__init__()
        self.proj = nn.Linear(d_model,d_out)
        self.logsftmx = nn.LogSoftmax(dim=2) #the 3rd dim is embedding size 
    def forward(self,x):
        return self.logsftmx(self.proj(x))

class convXL (nn.Module): 
    def __init__(self,pe,compressor,encoder,expander,generator):
        super(convXL,self).__init__()
        self.compressor = compressor
        self.encoder = encoder
        self.expander = expander
        self.generator = generator
        self.pe = pe

    def forward(self, src, src_mask=None,src_key_padding_mask=None):
        src2 = src.permute(1,2,0).contiguous()
        src3 = self.pe(self.compressor(src2).permute(2,0,1).contiguous())
        enc = self.encoder(src3,src_mask,src_key_padding_mask)
        expa = self.expander(enc.permute(1,2,0).contiguous())
        out = self.generator(expa.permute(2,0,1).contiguous())
        return out

def make_conv_encoder_model(d_in,d_model,d_out,nhead=8,nlayer=6,d_ff=2048,dropout=0.1):
    print("""Making conv_encoder model with Parameters:
                d_in={0}, d_model={1}, d_out={2}, nhead={3},
                nlayer={4}, d_ff={5}, dropout={6}""".format(d_in,
                d_model,d_out,nhead,nlayer,d_ff,dropout))
    encoder = Encoder(EncoderLayer(d_model,nhead,d_ff,dropout),nlayer)
    compressor = multiCompressor(d_in,d_model)
    expander = multiExpander(d_model,64) #changed Expander dim, make it option
    #expander = Expander(d_model,d_ff)
    #compressor = Compressor(d_in,d_model)
    pe = PositionalEncoding(d_model,dropout)
    generator = Generator(64,d_out)
    model = convXL(pe,compressor,encoder,expander,generator)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
