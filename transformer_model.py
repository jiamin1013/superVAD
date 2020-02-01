#!/usr/bin/env python3
#write my own transformer model to fascilitate flexibility
#based on "The Annotated Transformer"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from torch.autograd import Variable
import math, copy

def clone(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

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

class Generator(nn.Module):
    #Define standard linear+softmax generation step
    def __init__(self,d_model,d_out):
        super(Generator,self).__init__()
        self.proj = nn.Linear(d_model,d_out)
        self.logsftmx = nn.LogSoftmax(dim=2) #the 3rd dim is embedding size
    
    def forward(self,x):
        return self.logsftmx(self.proj(x))

class EncoderDecoder(nn.Module):
    #Base class for this and many other Encoder-Decoder models
    def __init__(self, encoder, decoder, tgt_embed, position_encode=None,generator=None):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_embed = tgt_embed #learn category to embeddings mapping
        self.pe = position_encode #positional encoding
        self.generator = generator #map embed space to category distribution
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        #src:(sSeq,batch,embed)
        #tgt:(tSeq,batch,embed)
        #src_mask:(sSeq,sSeq),additive
        #tgt_mask:(tSeq,tSeq),additive
        #memory_mask:(tSeq,sSeq),additive
        #src_key_padding_mask:(batch,sSeq),binary
        #tgt_key_padding_mask:(batch,tSeq),binary
        #memory_key_padding_mask:(batch,sSeq),binary
        #Take in and process masked src and target sequences.
        #output(tSeq,batch,embed)
        assert(src.size()[1]==tgt.size()[1]),"src batch size should be equal to target batch size (Note the 2nd dim of input and output is batch size)" 
        memory = self.encode(src, src_mask, src_key_padding_mask)
        out,_ = self.decode(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return self.generator(out)
    
    def encode(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.pe(src)
        return self.encoder(src2, src_mask, src_key_padding_mask)
    
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.pe(self.tgt_embed(tgt))
        return self.decoder(tgt2, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)

    def greedy_decode(self, memory, start_symbol, max_len,
                      memory_mask=None, memory_key_padding_mask=None):
        ys = torch.zeros((max_len,1),dtype=torch.long)
        ys_mask = self.generate_square_subsequent_mask(ys.size(0))
        next_ys = start_symbol
        atnw = []
        for i in range(max_len-1):
            ys[i,0] = next_ys 
            out,atw = self.decode(ys,memory,ys_mask,memory_mask,None,memory_key_padding_mask)
            prob = self.generator(out)[i,:,:]
            next_ys = torch.exp(prob.detach()).argmax(dim=1)#.unsqueeze(0)
#            atnw.append(atw.detach().numpy())
#            ys = torch.cat((ys,next_ys),dim=1)
        return ys,atw.detach().numpy()

    def generate_square_subsequent_mask(self,size):
        #Mask out subsequent positions
        attn_shape = np.empty([size,size])
        attn_shape.fill(float('-inf'))
        subsequent_mask = np.triu(attn_shape,k=1)
        #Fills lower triangular with 0.0
        return torch.as_tensor(subsequent_mask,dtype=torch.float32)

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
   
class Decoder(nn.Module):
    def __init__(self,decoder_layer,N,norm=None):
        super(Decoder,self).__init__()
        self.decoder_layers = clone(decoder_layer,N)
        self.num_layers = N
        self.norm = norm

    def forward(self,tgt,memory,tgt_mask=None,memory_mask=None,
                tgt_key_padding_mask=None,memory_key_padding_mask=None):
        output = tgt
        for i in range(self.num_layers):
            output,atw = self.decoder_layers[i](output,memory,tgt_mask=tgt_mask,memory_mask=memory_mask,tgt_key_padding_mask=tgt_key_padding_mask,memory_key_padding_mask=memory_key_padding_mask)
            if i==0:
                attnweight = atw #TODO, change later for plot attn

        if self.norm:
            output = self.norm(output)

        return output,attnweight

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
        tgt2,_ = self.self_attn(tgt,tgt,tgt,attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt+self.dropout1(tgt2))
        tgt2,atw = self.ende_attn(tgt,memory,memory,attn_mask=memory_mask,
                              key_padding_mask=memory_key_padding_mask) #change later for plot attn TODO
        tgt = self.norm2(tgt+self.dropout2(tgt2))
        #Assume Relu activation
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt+self.dropout3(tgt2))
        return tgt,atw

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=4000):
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

class Embeddings(nn.Module):
    def __init__(self, d_model, category):
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(category, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)

def make_txmodel(d_model=512,d_out=4,nhead=8,nlayer=6,d_ff=2048,dropout=0.1):
    encoder = Encoder(EncoderLayer(d_model,nhead,d_ff,dropout),nlayer)
    decoder = Decoder(DecoderLayer(d_model,nhead,d_ff,dropout),nlayer)
    tgt_embed = Embeddings(d_model,d_out)
    pe = PositionalEncoding(d_model,dropout)
    generator = Generator(d_model,d_out)
    model = EncoderDecoder(encoder,decoder,tgt_embed,pe,generator)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

#TODO, fix following functions
def train_tx_model(device,niter,optimizer,criterion,model,dataloaders,name="tx"):
    trainloader, devloader = dataloaders
    print("Training Transformer ...")
    for epoch in range(niter):
        print(epoch)
        loss_sum = 0
        for data in trainloader:
            running_loss = 0
            xseq,yseq,xlen,ylen = data
            xseq,yseq = xseq.to(device=device),yseq.to(device=device)

def lbl_to_tgtseq(lblseq,d_model,category=2):
    sz0,sz1 = lblseq.size()
    if sz0 < sz1:
        lblseq = torch.transpose(lblseq,0,1).contiguous()
        print("Transposed lblseq to (seqL,batch), halt program if \
        this is not by design")
    tgt1 = make_one_hot(lblseq,category)
    tgt,pad_sz = pad_vec_dim(tgt1,d_model)
#    tgt = pad_bos_eos(tgt2,pad_sz)
    tgt.requires_grad = True
    return tgt

def make_one_hot(lblseq,category=2):
    #assume lbl seq dim order (seqL,batch) 
    assert(lblseq.size()[1]<lblseq.size()[0])
    emb = nn.Embedding(category,category)
    emb.weight.data = torch.eye(category,dtype=torch.float32) 
    #this return a matrix of (seqL,batch,category_sz)
    return emb(lblseq).detach()

def pad_vec_dim(vec_in,d_model):
    #pad extra dimension with value 2 to vectors 
    #assume vec ordered dimension (seqL,batch,embed)
    if len(vec_in.size()) == 2:
        #some case our input label is (seqL,batch)
        vec_in = vec_in.unsqueeze(dim=2)
    seq_sz,b_sz,embed_sz = vec_in.size() 
    pad_sz = d_model - embed_sz
    pad_vec = torch.zeros((seq_sz,b_sz,pad_sz)) 
    vec_out = torch.cat((vec_in,pad_vec),dim=2).detach()
    return vec_out,pad_sz

def pad_bos_eos(seq_in,pad_sz=0):
    #pad bos and eos to labels 
    #assume input dimension to be ordered (seqL,batch,embed)
    seq_sz,b_sz,embed_sz = seq_in.size()
    bos = torch.full((1,b_sz,embed_sz),1000,dtype=seq_in.dtype)
    bos[:,:,embed_sz-pad_sz:] = 0
    eos = torch.full((1,b_sz,embed_sz),-1000,dtype=seq_in.dtype)
    eos[:,:,embed_sz-pad_sz:] = 0
    new_seq = torch.cat((bos,seq_in,eos),dim=0)
    return new_seq

#Note: Crossentropy loss does not require target sequence
#to be embedding size, but class prediction can be onehot
class LabelSmoothing(nn.Module):
    #Perform label smoothing to obtain a continuous
    #label distribution
    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        #self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
