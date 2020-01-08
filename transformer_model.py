#!/usr/bin/env python3
#write my own transformer model to fascilitate flexibility
#based on "The Annotated Transformer"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import math, copy

def clone(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Generator(nn.Module):
    #Define standard linear+softmax generation step
    def __init__(self,d_model,d_out):
        super(Generator,self).__init__()
        self.proj = nn.Linear(d_model,d_out)
    
    def forward(self,x):
        return F.log_softmax(self.proj(x),dim=-1)

class EncoderDecoder(nn.Module):
    #Base class for this and many other Encoder-Decoder models
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        #Take in and process masked src and target sequences.
        return self.decode(self.encode(src, src_mask), src_mask,tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Encoder(nn.Module):
    def __init__(self,encoder_layer,N):
        super(Encoder, self).__init__()
        #self.layers is a list of attn+FF sublayers
        self.encoder_layers = clone(encoder_layer,N)
        #define layer norm function, note size is intrinsic only to class EncoderLayer, it specifies the input size to layernorm
        self.num_layers = N
        self.norm = LayerNorm(encoder_layer.size)

    def forward(self,src,src_mask=None, src_key_padding_mask=None):
        #pass the input (and mask) through each layer
        output = src
        for i in range(self.num_layers):
            output = self.encoder_layers[i](output,src_mask = src_mask,
                            src_key_padding_mask = src_key_padding_mask) 
        if self.norm:
            output = self.norm(output)

        return output

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
   
class Decoder(nn.Module):
    def __init__(self,decoder_layer,N):
        self.layers = clone(decoder_layer,N)
        self.num_layers = N
        self.norm = norm

    def forward(self,tgt,memory,tgt_mask=None,memory_mask=None,
                tgt_key_padding_mask=None,memory_key_padding_mask=None):
        output = tgt
        for i in range(self.num_layers):
            output = self.layers[i](output,memory,tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)
        if self.norm:
            output = self.norm(output)

        return output

class DecoderLayer(nn.Module):
    def __init__(self,d_model,nhead,d_ff=2048,dropout=0.1):
        super(DecoderLayer,self).__init__()
        self.self_attn = MultiheadAttention(d_model,nhead,dropout)
        self.ende_attn = MultiheadAttention(d_model,nhead,dropout)
        #Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff,d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self,tgt,memory,tgt_mask=None,memory_mask=None,
                tgt_key_padding_mask=None,memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt,tgt,tgt,attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = self.norm1(tgt+self.dropout1(tgt2))
        tgt2 = self.ende_attn(tgt,memory,memory,attn_mask=memory_mask,
                              key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.norm2(tgt+self.dropout2(tgt2))
        #Assume Relu activation
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt+self.dropout3(tgt2))
        return tgt

def subsequent_mask(size):
    #Mask out subsequent positions
    attn_shape = np.empty(size,size)
    subsequent_mask = np.triu(attn_shape.fill(float('-inf'),k=1))
    #Fills lower triangular with 0.0
    return torch.as_tensor(subsequent_mask)










