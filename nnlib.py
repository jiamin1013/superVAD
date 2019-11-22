#!/usr/bin/env python3
#misc functions for nn vad experiment
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from sklearn.metrics import confusion_matrix,roc_curve,auc
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence, pack_sequence

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

    def __init__ (self,reco_list_sorted,mode,mean=0,transform=None):
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
        lbl = np.load("pyprep/{0}/{1}_lbl.npy".format(self.mode,reco)).squeeze()
        lbl = torch.as_tensor(1-lbl,dtype=torch.long)
        return feat,lbl

def data_load_datasets(data,global_mean):
    train,dev,test = data
    train_dataset = mfccDataset(train,mode="train",mean=global_mean)
    dev_dataset = mfccDataset(dev,mode="dev",mean=global_mean)
    test_dataset = mfccDataset(test,mode="test",mean=global_mean)
    return train_dataset,dev_dataset,test_dataset

def data_define_dataloader(datasets,batch_size,collate_func):
    train,dev,test = datasets
    train_loader = utils.DataLoader(train,batch_size=batch_size,
                            collate_fn=collate_func,num_workers=5)
    dev_loader = utils.DataLoader(dev,batch_size=batch_size,
                            collate_fn=collate_func,num_workers=5)
    test_loader = utils.DataLoader(test,batch_size=batch_size,
                            collate_fn=collate_func,num_workers=5)
    return train_loader,dev_loader,test_loader

def pad_collate(batch):
    #batch is [(seq1,lbl1),(seq2,lbl2),...(seqN,lblN)], N is batch size.
    (xx,yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    #padded sequence is B*T*x, T is longest sequence, x is featdim
    xx_pad = pad_sequence(xx,batch_first=True,padding_value=0)
    yy_pad = pad_sequence(yy,batch_first=True,padding_value=-100)
    return xx_pad,yy_pad,x_lens,y_lens

def pad_collate_conv(batch):
    (xx,yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    return

def get_num_pads(L,stride,kernel_size):
    L,s,ksz = [int(x) for x in (L,stride,kernel_size)]
    # pad to make sure all windows cover signal
    if (L-ksz)%s == 0:
        return 0 #no pad needed
    else:
        return int(np.ceil((L-ksz)/s)*s-(L-ksz)) #need a pad

def normalize_Data(data,mean):
    #subtract global mean
    print("Data Normalization ...")
    data =  np.subtract(data,mean)
    #length normalization to sqrt(feat-dim)
    feat_dim = np.shape(data)[1]
    l2_vect= np.sqrt(np.sum(np.square(data),axis=1,keepdims=True))
    data = np.divide(np.sqrt(feat_dim)*data,l2_vect)
    return data

def read_recolist(filename):
    recos = []
    with open(filename,'r') as f:
        for line in f:
            recos.append(line.strip())
    return recos

# -------- Training --------

def model_train_seq(device,niter,optimizer,criterion,model,dataloaders,name="lstm"):
    trainloader, devloader = dataloaders
    print("Training RNN ...") #TODO, make a summary of model
    for epoch in range(niter): #loop over the dataset multiple times
        print(epoch)
        loss_sum = 0
        for data in trainloader:
            running_loss = 0
            xseq,yseq,xlen,ylen = data
            xseq = xseq.to(device=device)
            yseq = yseq.to(device=device)
            model.train() #set train mode
            optimizer.zero_grad()
            yypad,_ = model((xseq,xlen))
            #swap the seq_len(2) and num_class(1) dims and keep batch(0) dim
            yyseq = yypad.permute((0,2,1)).contiguous()
            running_loss = criterion(yyseq,yseq)
            running_loss.backward()
            optimizer.step() #zero grad and update parameter
            print("running loss per batch: {0}".format(running_loss))
            loss_sum += running_loss.item()
        loss_avg = loss_sum/len(trainloader)
        print("epoch: {0}, training loss: {1:1.4f}".format(epoch,loss_avg))
        #save model and check validation each 100 epoch
        if (epoch+1)%100 == 0:
            model_eval_seq(model,devloader,criterion,mode="Validation")
            model_path = "results/rnn_params/tmp/{0}".format(name)
            torch.save(model.state_dict(),model_path)
    torch.save(model.state_dict(),"results/rnn_params/{0}".format(name))
    print("Training finished, model {0} is saved".format(name))
    return model

def model_eval_seq(device,model,loader,criterion,mode="Validation"):
    device = next(model.parameters()).device
    sftmax = nn.Softmax(dim=0)
    loss_sum,ya,yya = 0,[],[]
    for eval_data in loader:
        loss = 0
        xseq,yseq,xlen,ylen = eval_data
        xseq = xseq.to(device=device)
        yseq = yseq.to(device=device)
        model.eval() #set eval mode
        yypad,_ = model((xseq,xlen))
        yyseq = yypad.permute((0,2,1)).contiguous()
        loss_sum += criterion(yyseq,yseq).item()
        #loop batch
        for bi in range(yyseq.size()[0]):
            chop = ylen[bi]
            #probability of 0-th class, namely probability of voiced
            yy = sftmax(yyseq[bi,:,:chop]).squeeze()[0,:]
            y = yseq[bi,:chop]
            ya.append(y); yya.append(yy)
    ya = 1-torch.cat(ya,dim=0).cpu().detach().numpy()
    yya = torch.cat(yya,dim=0).cpu().detach().numpy()
    #np.save("results/val_run2_ly2_ep500",(yya,ya))
    fpr,tpr,thresholds = roc_curve(ya,yya)
    optix = np.absolute(tpr-(1-fpr)-0).argsort()[0]
    roc_auc = auc(fpr,tpr)
    print("{0} loss: {1:2.2f}".format(mode,loss_sum/len(loader)))
    print("{0} AUC: {1:2.2f}".format(mode,roc_auc))
    print("{0} frame recall rate (voiced): {1:2.2f}%".format(mode,tpr[optix]*100))
    print("{0} frame false alarm rate (voiced): {1:2.2f}%".format(mode,fpr[optix]*100))

def cal_diff(yy,y):
    assert(len(yy)==len(y))
    mask = y==1 #positive class
    t_pos = torch.sum(yy[mask]==y[mask]).numpy() #num of true positives
    t_neg = torch.sum(yy[~mask]==y[~mask]).numpy() #num of true negative
    f_pos = torch.sum(yy[~mask]!=y[~mask]).numpy() #num of false positive
    f_neg = torch.sum(yy[mask]!=y[mask]).numpy() #num of false negative
    return t_pos,t_neg,f_pos,f_neg

# -------- neural networks -----------

class LSTMNet(nn.Module):
    def __init__(self,dim_input,dim_output,dim_hidden,num_layers=1):
        super(LSTMNet,self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.i2h = nn.LSTM(self.dim_input,self.dim_hidden,num_layers=num_layers)
        self.logits_fc = nn.Linear(self.dim_hidden,self.dim_output)

    def forward(self,xx,hidden=None):
        xseq,xlen = xx
        x = pack_padded_sequence(xseq,xlen,batch_first=True,
                                        enforce_sorted=False)
        packed_out,last_hidden = self.i2h(x,hidden)
        padded_out,_ = pad_packed_sequence(packed_out,batch_first=True,
                                           padding_value=0.0)
        padded_y = self.logits_fc(padded_out)
        return padded_y,last_hidden

    def initHidden(self):
        return torch.zeros(1,self.dim_hidden)

class ConvLSTM(nn.Module):
    def __init__(self,dim_input,dim_output,dim_hidden):
        super(ConvLSTM,self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.conv1d = nn.Conv1d(in_channels=dim_input,out_channels=128,
                                kernel_size=64,stride=1,padding=0)
        self.lstm = nn.LSTM(128,128,num_layers=1)
        self.convT1d = nn.ConvTranspose1d(in_channels=128,out_channels=64,
                                kernel_size=64,stride=1,padding=0)
        self.affine = nn.Linear(64,32)
        self.logits_fc = nn.Linear(32,dim_output)

    def forward(self,x,hidden=None):
        packed_out,last_hidden = self.i2h(x,hidden)
        padded_out,_ = pad_packed_sequence(packed_out,batch_first=True,padding_value=0.0)
        padded_output = self.logits_fc(padded_out)
        return padded_output,last_hidden

    def initHidden(self):
        return torch.zeros(1,self.dim_hidden)

#misc

