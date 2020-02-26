#!/usr/bin/env python3
#speech sequence tagger experiment
#e.g. VAD, speaker-type detection

#basic template code for data preprocessing for transformer encoder 
#2020 Jiamin Xie

import torch
import numpy as np
import torch.utils.data as utils
from torch.nn.utils.rnn import pad_sequence

#----- data preprocessing ----
class mfccDataset(utils.Dataset):
    # args:
    # reco_list_sorted: reco_list sorted by increasing length
    # mode: train/dev/test dataset to be prepared
    # mean: value to be subtracted from features
    # transform: any transformation applied to the features
    # note:
    # when voice/silence is read from kaldi, 1 denotes voiced frame,
    # and 0 denotes unvoiced frame, but for the nn training purpose,
    # since CrossEntropy loss is used, the class label is defined as:
    # VOICED -> 0-th class, SILENCE -> 1-st class, you can think 
    # i-th output node is the probability of i-th class occuring
    # so, we prepare our lbl data this way 

    def __init__ (self,reco_list_sorted,mode,mean=0,transform=None,
                  chunk_size=6416):
        self.reco_list = reco_list_sorted #reco_list sorted by increasing length
        self.miu = mean
        self.mode = mode
        self.transform = transform
        self.chop_size = chunk_size

    def __len__ (self):
        return len(self.reco_list)

    def __getitem__(self,idx):
        reco = self.reco_list[idx]
        #mfcc feats, assume normalized by provided mean; chop data
        feat = np.load("pyprep/{0}/{1}_mfcc.npy".format(self.mode,reco))
        feat = chop_data(feat,self.chop_size) #will discard residual
        feat = torch.from_numpy(feat-self.miu)
        #frame labels, for vad, we offset label by 1 to indicate class
        lbl = np.load("pyprep/{0}/{1}_lbl.npy".format(self.mode,reco)).squeeze()
        lbl = chop_data(lbl,self.chop_size) #will discard residual
        lbl = torch.as_tensor(1-lbl,dtype=torch.long)
        assert (feat.size(0)==lbl.size(0)),"Feat sequence length {0}\
        not equal to label sequence length {1}".format(feat.size(0),
                                                       lbl.size(0))
        return feat,lbl

def chop_data(data,chunk_size):
    #data is [SeqL, featdim]
    end = data.shape[0]//chunk_size*chunk_size
    if len(data.shape) != 1:
        chopped = np.split(data[:end,:],int(end/chunk_size),axis=0)
    else:
        chopped = np.split(data[:end],int(end/chunk_size),axis=0)
    return np.stack(chopped,axis=0) #data is [chunks,chunkL,featdim]

def data_load_datasets(data,global_mean):
    train,dev,test = data #lists of train, dev, test recordings
    train_dataset = mfccDataset(train,mode="train",mean=global_mean)
    dev_dataset = mfccDataset(dev,mode="dev",mean=global_mean)
    test_dataset = mfccDataset(test,mode="test",mean=global_mean)
    return train_dataset,dev_dataset,test_dataset

#--- data loader part (sample retrived by NN) ---
def data_define_dataloader(datasets,batch_size,collate_func=None):
    train,dev,test = datasets
    train_loader = utils.DataLoader(train,batch_size=batch_size,
                            collate_fn=collate_func,num_workers=5)
    dev_loader = utils.DataLoader(dev,batch_size=batch_size,
                            collate_fn=collate_func,num_workers=5)
    test_loader = utils.DataLoader(test,batch_size=batch_size,
                            collate_fn=collate_func,num_workers=5)
    return train_loader,dev_loader,test_loader

#--- template collate function for rnn padding ---
def pad_collate(batch):
    #collate function for tx encoder
    #batch is [(seq1,lbl1),(seq2,lbl2),...(seqN,lblN)], N is batch size.
    (xx,yy) = zip(*batch)
    xx_batch = torch.cat(xx).permute(1,0,2).contiguous()
    yy_batch = torch.cat(yy).permute(1,0).contiguous()
    #data is in [chunkL,bsize,dim]
    return xx_batch,yy_batch 

def pad_collate_conv(batch):
    (xx,yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    xx_new,yy_new = [],[]
    for i in range(len(xx)):
        x_conv,y_conv = pad_convseq((xx[i],yy[i]))
        xx_new.append(x_conv)
        yy_new.append(y_conv)
    xx_pad = pad_sequence(tuple(xx_new),batch_first=True,padding_value=0)
    yy_pad = pad_sequence(tuple(yy_new),batch_first=True,padding_value=-100)
    return xx_pad, yy_pad, x_lens, y_lens

def pad_convseq(seqs):
    xx,yy = seqs
    stride, kernel_size = 32, 64 #change with hyperparam
    num_padx = get_num_pads(len(xx),stride,kernel_size)
    num_pady = get_num_pads(len(yy),stride,kernel_size)
    xpad = torch.tensor((),dtype=xx.dtype).new_full((num_padx,xx.size()[1]),0)
    ypad = torch.tensor((),dtype=yy.dtype).new_full((num_pady,),-100)
    xx = torch.cat([xx,xpad],dim=0)
    yy = torch.cat([yy,ypad],dim=0)
    return xx , yy

def get_num_pads(L,stride,kernel_size):
    L,s,ksz = [int(x) for x in (L,stride,kernel_size)]
    # pad to make sure all windows cover signal
    if (L-ksz)%s == 0:
        return 0 #no pad needed
    else:
        return int(np.ceil((L-ksz)/s)*s-(L-ksz)) #need a pad

def get_num_conv_out(newL,kernel_size,stride):
    return int((newL-kernel_size)/stride)+1

def generate_padding_mask(seqlens):
    #seqlens is [seq1len,seq2len,seq3len,...,seqNlen], N is batch size.
    bsize = len(seqlens)
    seqL = max(seqlens) 
    #mask with True value will be masked, dtype=byteTensor, (batch,seqL)
    mask = torch.empty(bsize,seqL).fill_(False).byte()
    for bidx in range(bsize): 
        #subscripting is inclusive
        mask[bidx,seqlens[bidx]:] = True #mask padded elements
    return mask


