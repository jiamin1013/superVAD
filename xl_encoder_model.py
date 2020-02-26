#!/usr/bin/env python3
#speech sequence tagger experiment
#e.g. VAD, speaker-type detection

#basic template code for tx encoder models 
#2020 Jiamin Xie

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math,copy
#from torch.nn.utils.rnn import pack_padded_sequence
#from sklearn.metrics import confusion_matrix,roc_curve,auc

#--- functions
def clone(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

#--- tx encoder specific
class Generator(nn.Module):
    #Define standard linear+softmax generation step
    def __init__(self,d_model,d_out):
        super(Generator,self).__init__()
        self.proj = nn.Linear(d_model,d_out)
        self.logsftmx = nn.LogSoftmax(dim=2) #the 3rd dim is embedding size 
    def forward(self,x):
        return self.logsftmx(self.proj(x))

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

#--- Tx Encoder General
class Encoder(nn.Module):
    def __init__(self,encoder_layer,N,norm=None):
        super(Encoder, self).__init__()
        #self.layers is a list of attn+FF sublayers
        self.encoder_layers = clone(encoder_layer,N)
        self.num_layers = N
        self.norm = norm

    def forward(self,src,src_mask=None, src_key_padding_mask=None):
        #pass the input (and mask) through each layer
        output = src
        for i in range(self.num_layers):
            output = self.encoder_layers[i](output,src_mask=src_mask,
                                 src_key_padding_mask=src_key_padding_mask)
        if self.norm:
            output = self.norm(output)

        return output

class EncoderLayer(nn.Module):
    def __init__(self,d_model,nhead,d_ff=2048,dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model,nhead,dropout=dropout)
        #Position-wise Feedfoward model (each vector passes thru ff)
        self.linear1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.size = d_model

    def forward(self,src,src_mask=None,src_key_padding_mask=None):
        src2 = self.self_attn(src,src,src,attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = self.norm1(src + self.dropout1(src2)) #norm(residual)
        #Assume we use relu as activation
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src

class MultiheadAttention(nn.Module):
    #This is quite troublesome to implement for now,
    #Since multiheadattention is used in torch 1.1.0, we will call 
    #it instead, TODO, implement multihead attn from scratch
    def __init__(self,d_model,nhead,dropout=0.1):
        super(MultiheadAttention,self).__init__()
        self.attn = nn.MultiheadAttention(d_model,nhead,dropout)

    def forward(self,query,key,value,attn_mask=None,key_padding_mask=None):
        return self.attn(query,key,value,
                         key_padding_mask=key_padding_mask,
                         attn_mask=attn_mask)

class LayerNorm(nn.Module):
    #Note the difference between this norm and pytorch norm is that
    #this one use unbiased estimator for variance/std estimation,
    #the pytorch version uses biased estimator, difference is denominator
    #1/(H-1) unbiased vs. 1/H biased
    def __init__(self,layer_size,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.layer_size = layer_size
        self.gain = nn.Parameter(torch.ones(layer_size))
        self.bias = nn.Parameter(torch.zeros(layer_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(-1,keepdim=True) #unbiased estimation
        return self.gain*(x-mean)/(std+self.eps)+self.bias

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=8000):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0.,max_len).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0.,d_model,2)/ \
                             d_model*math.log(10000.0))
        #broadcast position to div_terms
        pe[:,0::2] = torch.sin(position * div_term) #even indicies
        if d_model % 2 != 0:
            pe[:,1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1) #batch dimension
        self.register_buffer('pe',pe)

    def forward(self,x):
        #x dim is SeqL, Batch, Embed
        #add positional encoding to embedding
        x = x + Variable(self.pe[:x.size(0),:,:], requires_grad = False)
        return self.dropout(x)

#TX encoder Model
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
        src3 = self.compressor(src2).permute(2,0,1).contiguous()
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
