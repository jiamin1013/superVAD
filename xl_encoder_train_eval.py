#!/usr/bin/env python3
#speech sequence tagger experiment
#e.g. VAD, speaker-type detection

#template code to train/evaluate tx encoder model
#2020 Jiamin Xie

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from sklearn.metrics import confusion_matrix,roc_curve,auc

#Training tx encoder
def model_train(device,niter,optimizer,criterion,model,dataloaders,name="tx_model"):
    trainloader, devloader = dataloaders
    print("Training ConvXL ...")
    print(torch.__version__)
    losses = []
    t = time.time()
    for epoch in range(niter):
        print(epoch)
        loss_sum = 0
        for data in trainloader:
            running_loss = 0
            x_batch,y_batch = data
            #print(x_batch.size())
            x_batch.to(device=device)
            y_batch.to(device=device)
            #src: xseq, tgt: yseq (seqL,bsize,embedsz)
            model.train() #set train mode
            optimizer.zero_grad()
            yy_batch = model(x_batch,src_mask=None,src_key_padding_mask=None)
            #permute yypad to have (batch,class,seqL) because KLDivLoss is batch first
            yyseq = yy_batch.permute(1,2,0).contiguous()
            running_loss = criterion(yyseq,y_batch)/x_batch.size(0)
            running_loss.backward()
            optimizer.step()
            print("running loss per batch: {0}".format(running_loss))
            loss_sum += running_loss.item()
        loss_avg = loss_sum/len(trainloader)
        losses.append(loss_avg)
        print("epoch: {0}, training loss: {1:1.4f}".format(epoch,loss_avg))
        #save model and check validation each 100 epoch
        if (epoch+1)%100 == 0:
            #TODO: save model properly, and write model eval function
            model_path = "results/rnn_params/tmp/{0}_{1}".format(name,epoch)
            torch.save(model.state_dict(),model_path)
            #validate
            model_eval(model,devloader,criterion,mode="Validation")
    elapsed = time.time()-t
    print("{} time elapsed.".format(elapsed))
    torch.save(model.state_dict(),"results/rnn_params/{0}".format(name))
    print("Training finished, model {0} is saved".format(name))
    return model,losses

def model_eval(model,loader,criterion,mode="Validation"):
    device = next(model.parameters()).device
    print("{}...".format(mode))
    loss_sum,ya,yya = 0,[],[]
    for eval_data in loader:
        loss = 0
        xseq,yseq = eval_data #here we assume a batch is from one recording
        xseq.to(device=device) #dim=(chunkL,bsize,embedsz)
        yseq.to(device=device) #dim=(chunkL,bsize)
        model.eval()
        yyseq1 = model(xseq,src_mask=None,src_key_padding_mask=None) #yyseq1 is log distribution of labels, dim=(chunkL,bsize,embedsz)
        yyseq2 = yyseq1.permute(1,2,0).contiguous() #dim=(bsize,embedsz,chunkL)
        loss_sum += criterion(yyseq2,yseq).item()/xseq.size(0)
        #Measure Performance
        yyseq = torch.argmax(yyseq1,dim=2).cpu().detach().numpy() #(chunkL,bsize)
        #yseq is dim=(chunkL,bsize), 0 is voiced, 1 is unvoiced
        yseq = yseq.cpu().detach().numpy()
        yyseq = yyseq.T.reshape(-1)
        yseq = yseq.T.reshape(-1)
        ya.append(yseq)
        yya.append(yyseq)
    ya = np.concatenate(ya,axis=0)
    yya = np.concatenate(yya,axis=0)
    t_pos,t_neg,f_pos,f_neg = cal_diff(yya,ya,positive=0)
    #Check calculation for frame recall and false alarm
    print("{0} loss: {1:2.2f}".format(mode,loss_sum/len(loader)))
#    print("{0} AUC")
    print("{0} frame recall rate (voiced): {1:2.2f}%".format(mode,t_pos/(t_pos+f_neg)*100))
    print("{0} frame false alarm rate (voiced): {1:2.2f}%".format(mode,f_pos/(f_pos+t_neg)*100))

#For model loss and evaluation
class LabelSmoothing(nn.Module):
    #Perform label smoothing to obtain a continuous label distribution
    def __init__(self, size, smoothing=0.0,pad_val=-100,add_bos=False):
        super(LabelSmoothing, self).__init__()
        #KLDiv = sum_x P(x)*log(P(x)/Q(x)), note reduction = batchmean,
        #which sums up rest of dimensions as support size and divide by batch size (1st dim) 
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.padding_val = pad_val
        self.true_dist = None
        self.add_bos = add_bos

    def forward(self, x, gold, padding_mask=None):
        #gold, gold label dim=(seqL,bsz), x is predicted dim=(batch,class,seqL) 
        gold_t = gold.transpose(0,1).contiguous() #Transpose gold label
        #Hack, need Check later #TODO
        gold_t[gold_t==-100] = 0 #To remove wrong index, padding index will be masked in the end, so we don't need to worry
        assert x.size(1) == self.size 
        assert x.size(0) == gold_t.size(0)
        #Construct gold distribution from gold label
        true_dist = torch.empty(x.size()) #dim=(batch,class,seqL) 
        if not self.add_bos:
            smooth_val = self.smoothing/(self.size-1) #ignore the hot label
        else:
            smooth_val = self.smoothing/(self.size-2) #ignore bos probs
        #scatter gold label position by confidence, and fill the rest with smooth
        true_dist.fill_(smooth_val)
        true_dist.scatter_(1, gold_t.data.unsqueeze(1), self.confidence)
#        true_dist.masked_fill_(padding_mask.unsqueeze(1),0) #True value will be masked
#       DEBUG HERE TODO
        true_dist.detach()
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

def cal_diff(yy,y,positive=0):
    #yy is prediction, y is truth
    assert(len(yy)==len(y))
    mask = y==positive #positive class
    t_pos = np.sum(yy[mask]==y[mask]).numpy() #num of true positives
    t_neg = np.sum(yy[~mask]==y[~mask]).numpy() #num of true negative
    f_pos = np.sum(yy[~mask]!=y[~mask]).numpy() #num of false positive
    f_neg = np.sum(yy[mask]!=y[mask]).numpy() #num of false negative
    #t_pos = torch.sum(yy[mask]==y[mask]).numpy() #num of true positives
    #t_neg = torch.sum(yy[~mask]==y[~mask]).numpy() #num of true negative
    #f_pos = torch.sum(yy[~mask]!=y[~mask]).numpy() #num of false positive
    #f_neg = torch.sum(yy[mask]!=y[mask]).numpy() #num of false negative
    return t_pos,t_neg,f_pos,f_neg
