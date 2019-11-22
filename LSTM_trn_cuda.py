#! /usr/bin/env python3
# Copyright  2019  Jiamin Xie
#
# --- imports ---
import numpy as np
import random
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as utils
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence, pack_sequence
import kaldi_io as kio
import subprocess
import sys
import time as T
import argparse
from sklearn.metrics import confusion_matrix,roc_curve,auc
#Setting
#np.set_printoptions(threshold=sys.maxsize)

#Input Assumption
#1) directory which contains train, dev, test partitions,
#   within each there is <reco>_mfcc.npy and <reco>_lbl.npy
#   which shows labeled frame-level features, required file
#   will be the concatenated version of all recordings, in 
#   format as <partition>_all_mfcc.npy and <partition>_all_lbl.npy

def get_args():
    parser = argparse.ArgumentParser(description = """na""")
    parser.add_argument("--norm", type=str2bool, default=True,
                        help="""Guassianize (normalize)
                        subsegment features, default is 1""")
    parser.add_argument("--pca", type=bool, default=False,
                        help="""whether dim reduce""")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="""learning rate""")
    parser.add_argument("--H", type=int, default=12,
                        help="""hidden layer size""")
    parser.add_argument("--niter", type=int, default=500,
                        help="""epoch number""")
    parser.add_argument("--disable_cuda",type=str2bool,default=True,
                        help = """disable using GPU""")
    args = parser.parse_args()
    return args

#for debug purpose
def pr(x):
    print(x)
    input()

class mfccDataset(utils.Dataset):
    def __init__ (self,reco_list_sorted,mode="train",mean=0,transform=None):
        self.reco_list = reco_list_sorted #reco_list sorted by increasing length
        self.miu = mean
        self.mode = mode
        self.transform = transform

    def __len__ (self):
        return len(self.reco_list)

    def __getitem__(self,idx):
        reco = self.reco_list[idx]
        feat = np.load("pyprep/{0}/{1}_mfcc.npy".format(self.mode,reco))-self.miu
        feat = torch.from_numpy(feat)
        #make sure array has only one dim
        lbl = np.load("pyprep/{0}/{1}_lbl.npy".format(self.mode,reco)).squeeze()
        #note to be compatible with CrossEntropy loss, we convert voiced/unvoiced
        #decision 1/0, to classes label 0 and 1, so nueral network will have two
        #output nodes, with the upper node denoting posterior of voiced frame and 
        #lower node reprsents posterior of unvoiced frame
        lbl = torch.as_tensor(1-lbl,dtype=torch.long)
        return feat,lbl

def main_init():
    #initialization of main, usually instantiate required files
    trn_utts = read_recolist("pyprep/train/reco_list_sorted")
    dev_utts = read_recolist("pyprep/dev/reco_list_sorted")
    tst_utts = read_recolist("pyprep/test/reco_list_sorted")
    return trn_utts,dev_utts,tst_utts

def read_recolist(filename):
    recos = []
    with open(filename,'r') as f:
        for line in f:
            recos.append(line.strip())
    return recos

def pad_collate(batch):
    #batch is [(seq1,lbl1),(seq2,lbl2),...(seqN,lblN)], N is batch size.
    (xx,yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    #padded sequence is B*T*x, T is longest sequence, x is featdim
    xx_pad = pad_sequence(xx,batch_first=True,padding_value=0)
    yy_pad = pad_sequence(yy,batch_first=True,padding_value=-100)
    return xx_pad,yy_pad,x_lens,y_lens

def main():
    args = get_args()
    np.random.seed(0)
    torch.manual_seed(0)
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    trn_utts,dev_utts,tst_utts = main_init()
    #normalization
    glb_m = np.mean(np.load("pyprep/train/train_all_mfcc.npy"),axis=0,keepdims=True)
    #setup hyperparameters 
    din,dout,bsize = glb_m.shape[1],2,10
    glb_m = glb_m if args.norm else 0
    lr,dhid1,niter = args.lr,args.H,args.niter
    trnds = mfccDataset(trn_utts,mode="train",mean=glb_m)
    trnldr = utils.DataLoader(trnds,batch_size=bsize,collate_fn=pad_collate,num_workers=5)
    devds = mfccDataset(dev_utts,mode="dev",mean=glb_m)
    devldr = utils.DataLoader(devds,batch_size=bsize,collate_fn=pad_collate,num_workers=5)
    tstds = mfccDataset(tst_utts,mode="test",mean=glb_m)
    tstldr = utils.DataLoader(tstds,batch_size=bsize,collate_fn=pad_collate,num_workers=5)

    #configure net
    lstm = LSTMNet(din,dout,dhid1,layers=2,bidirect=True).to(device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100) #we use cross entropy with two classes
    optimizer = optim.SGD(lstm.parameters(),lr=lr,momentum=0.9)
    #load previous trained progress
#    lstm.load_state_dict(torch.load("results/rnn_params/run0_ly1_ep100_lr0.001_h12_bilstm"))
    print("Training RNN ...")
    for epoch in range(niter): #loop over the dataset multiple times
        print(epoch)
        loss_sum = 0
        for data in trnldr:
            celoss = 0
            xsq,ysq,xln,yln = data
            xsq,ysq = xsq.cuda(),ysq.cuda()
            xpk = pack_padded_sequence(xsq,xln,batch_first=True,enforce_sorted=False)
            lstm.train() #set train mode
            optimizer.zero_grad()
            yypk,_ = lstm(xpk)
            # we want to swap the seq_len(2) and num_class(1) dims, also keep batch dim (0)
            yysq = yypk.permute((0,2,1)).contiguous()
            # ysq is in correct dimens, i.e. (batch(N),K(seq_len))
            celoss = criterion(yysq,ysq)
            celoss.backward()
            optimizer.step() #zero grad and update parameter
            print("running loss: {0}".format(celoss))
            loss_sum += celoss.item()
        verbose(epoch,loss_sum/len(trnldr),"training")
#    torch.save(lstm.state_dict(),"results/rnn_params/run2_ly2_ep500_lr0.001_h12_lstm")
#    torch.save(lstm.state_dict(),"results/rnn_params/run0_normalize_ly1_ep100_lr0.0005_h12_lstm")
#    torch.save(lstm.state_dict(),"results/rnn_params/run0_nomean_ly1_ep100_lr0.0005_h12_lstm")
    torch.save(lstm.state_dict(),"results/rnn_params/run1_ly2_ep500_lr0.001_h12_bilstm")
#---comment above if load pre-trained model
    #validate/eval
    sftmax = nn.Softmax(dim=0)
#    for threshold in np.arange(0,1.1,0.1):
    loss_sum,ya,yya = 0,[],[]
    for data in devldr:
        loss,error = 0,0
        xsq,ysq,xln,yln = data
        xsq,ysq = xsq.cuda(),ysq.cuda()
        xpk = pack_padded_sequence(xsq,xln,batch_first=True,enforce_sorted=False)
        lstm.eval() #set eval mode
        yypad,_ = lstm(xpk)
        yysq = yypad.permute((0,2,1)).contiguous()
        #loop batch
        for bi in range(yysq.size()[0]):
            chop = yln[bi]
            #probability of 0-th class, namely probability of voiced
            yy = sftmax(yysq[bi,:,:chop]).squeeze()[0,:]
            y = ysq[bi,:chop]
            ya.append(y); yya.append(yy)
            #celoss = criterion(yysq,ysq)
            #print("running loss: {0}".format(celoss))
            #loss_sum += celoss.item()
    ya = 1-torch.cat(ya,dim=0).cpu().detach().numpy()
    yya = torch.cat(yya,dim=0).cpu().detach().numpy()
    fpr,tpr,thresholds = roc_curve(ya,yya)
    optix = np.absolute(tpr-(1-fpr)-0).argsort()[0]
    roc_auc = auc(fpr,tpr)
#    np.save("results/val_run2_ly2_ep500",(yya,1-ya))
    print("Val AUC: {0:2.2f}".format(roc_auc))
    print("Val frame recall rate (voiced) at EER: {0:2.2f}%".format(tpr[optix]*100))
    print("Val frame false alarm rate (voiced) at EER: {0:2.2f}%".format(fpr[optix]*100))
#    verbose("validation",loss_sum/len(devldr),"development")
    #Testing
    loss_sum,ya,yya = 0,[],[]
    for data in tstldr:
        loss,error = 0,0
        xsq,ysq,xln,yln = data
        xsq,ysq = xsq.cuda(),ysq.cuda()
        xpk = pack_padded_sequence(xsq,xln,batch_first=True,enforce_sorted=False)
        lstm.eval() #set eval mode
        yypad,_ = lstm(xpk)
        yysq = yypad.permute((0,2,1)).contiguous()
        #loop batch
        for bi in range(yysq.size()[0]):
            chop = yln[bi]
            #probability of 0-th class, namely probability of voiced
            yy = sftmax(yysq[bi,:,:chop]).squeeze()[0,:]
            y = ysq[bi,:chop]
            ya.append(y); yya.append(yy)
            #celoss = criterion(yysq,ysq)
            #print("running loss: {0}".format(celoss))
            #loss_sum += celoss.item()
    ya = 1-torch.cat(ya,dim=0).cpu().detach().numpy()
    yya = torch.cat(yya,dim=0).cpu().detach().numpy()
    #np.save("results/tst_run2_ly2_ep500",(yya.detach().numpy(),ya.detach().numpy()))
    fpr,tpr,thresholds = roc_curve(ya,yya)
    optix = np.absolute(tpr-(1-fpr)-0).argsort()[0]
    roc_auc = auc(fpr,tpr)
    print("Test AUC: {0:2.2f}".format(roc_auc))
    print("Test frame recall rate (voiced): {0:2.2f}%".format(tpr[optix]*100))
    print("Test frame false alarm rate (voiced): {0:2.2f}%".format(fpr[optix]*100))
#    verbose("Test",loss_sum/len(tstldr),"test")


def cal_diff(yy,y):
    assert(len(yy)==len(y))
    mask = y==1 #positive class
    t_pos = torch.sum(yy[mask]==y[mask]).numpy() #num of true positives
    t_neg = torch.sum(yy[~mask]==y[~mask]).numpy() #num of true negative
    f_pos = torch.sum(yy[~mask]!=y[~mask]).numpy() #num of false positive
    f_neg = torch.sum(yy[mask]!=y[mask]).numpy() #num of false negative
    return t_pos,t_neg,f_pos,f_neg


#Functions
# --NN models--
class LSTMNet(nn.Module):
    def __init__(self,d_in,d_out,h1,layers=1,bidirect=False):
        super(LSTMNet,self).__init__()
        self.num_direction = 2 if bidirect else 1
        self.hidden_size = h1
        self.input_size = d_in
        self.i2h = nn.LSTM(self.input_size,self.hidden_size,num_layers=layers,bidirectional=bidirect)
        self.logits_fc = nn.Linear(self.hidden_size*self.num_direction,d_out)

    def forward(self,x,hidden=None):
        packed_out,last_hidden = self.i2h(x,hidden)
        padded_out,_ = pad_packed_sequence(packed_out,batch_first=True,padding_value=0.0)
        output = self.logits_fc(padded_out)
        return output,last_hidden

# we use default hidden vectors (which are all zeros)
#    def initHidden(self):
#        return torch.zeros(1,self.hidden_size)

def est_w(lbl):
    lb = np.load(lbl)
    w = torch.tensor([1/np.sum(lb==i) for i in np.unique(lb)])
    return w

def printCM(cm):
    print("--- Un-normalized confusion matrix ---")
    print(['T|P',"0","1"])
    yax = ["0","1"]
    cf = np.insert(cm,0,yax,axis=1)
    print(cf)
    print("--- Normalized confusion matrix ---")
    cm = cm.astype('float')/cm.sum(axis=1,keepdims=True)
    yax = ["0","1"]
    print(["T|P","0","1"])
    cm = np.insert(cm,0,yax,axis=1)
    print(np.around(cm,decimals=3))

def verbose(epoch,loss,mode="train"):
    print("epoch: {0}, {1} loss: {2:1.4f}".format(epoch,mode,loss))

def normalize_Data(data,mean):
    #subtract global mean
    print("Data Normalization ...")
    data =  np.subtract(data,mean)
    #length normalization to sqrt(feat-dim)
    feat_dim = np.shape(data)[1]
    l2_vect= np.sqrt(np.sum(np.square(data),axis=1,keepdims=True))
    data = np.divide(np.sqrt(feat_dim)*data,l2_vect)
    return data

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__=="__main__":
    main()


