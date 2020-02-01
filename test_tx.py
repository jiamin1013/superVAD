#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from transformer_model import *
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")

def load_data(category,d_model):
    n = 5000
    lbl = torch.as_tensor(1-np.load("sample/sample_lbl.npy"),dtype=torch.long)
    lbl = torch.transpose(lbl,0,1)[n:n+1000,:]
    src = torch.as_tensor(np.load("sample/sample_mfcc.npy"),dtype=torch.float32).unsqueeze(1) #seqL,batch,d_model
    src = src[n:n+1000,:,:]
    bos = torch.as_tensor([[2]],dtype=torch.long)
    eos = torch.as_tensor([[3]],dtype=torch.long)
    lbl = torch.cat((bos,lbl,eos),dim=0)
    tgt = lbl[:-1]
    return src, tgt, lbl

def gen_random_data(seqL,d_model,bsz=1):
    lbl = torch.as_tensor(torch.randint(2,(seqL,bsz)),dtype=torch.long)
    bos = torch.as_tensor([[2]],dtype=torch.long) #<bos> = 3rd class, valued 2
    eos = torch.as_tensor([[3]],dtype=torch.long) #<eos> = 4th class, valued 3
    lbl = torch.cat((bos,lbl,eos),dim=0) #seqL,batch
#    tgt = lbl_to_tgtseq(lbl,d_model,category+2) #unnecessary as we will learn target embeddings
    tgt = lbl[:-1]
    src = torch.randint(-500,500,(seqL,1,d_model))
    src = torch.as_tensor(src,dtype=torch.float32)
    return src, tgt, lbl

def gen_data(category,seqL,bsz,d_model,fillvalue=500.):
    featmap = nn.Embedding(category,d_model)
    wt = torch.empty((category,d_model))
    wt[0,:] = torch.full((1,d_model),-fillvalue)
    wt[1,:] = torch.full((1,d_model),fillvalue)
    wt[2,:] = torch.full((1,d_model),2.)
    wt[3,:] = torch.full((1,d_model),3.)
    featmap.weight = torch.nn.Parameter(wt,requires_grad=False)
    lbl = torch.full((seqL+2,bsz),1,dtype=torch.long)
    lbl[0::3,:] = 0
    lbl[0,:] = 2 #bos
    lbl[-1,:] = 3 #eos
    tgt = lbl[:-1]
    src = featmap(lbl) 
    return src, tgt, lbl

#Run a simple copying example
#Current result is not good, need analysis
d_model = 13
nhead = 1
category = 4 #including bos, eos
nepoch = 1000
nlayer = 1
d_ff = 64
seqL = 25
bsz = 1
lr = 0.0001
mom = 0.9

torch.manual_seed(0)
np.random.seed(0)

model = make_txmodel(d_model=d_model,d_out=category,nlayer=nlayer,
                   d_ff=d_ff, nhead=nhead,dropout=0.)
src,tgt,lbl = gen_data(category,seqL,bsz,d_model,500.)
#src,tgt,lbl = gen_random_data(seqL,d_model)
#src,tgt,lbl = load_data(category,d_model)
tgt_mask = model.generate_square_subsequent_mask(len(tgt))
#
#criterion = nn.CrossEntropyLoss()
criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.SGD(model.parameters(),lr=lr,momentum=mom)

loss_sum = []

#a = model.encode(src,None,None)
#input()
#plt.subplot(211)
#plt.plot(src[:,0,0].numpy())
#plt.subplot(212)
#plt.plot(a[:,0,0].detach().numpy())
#plt.show()
#input()

for epoch in range(nepoch):
    print(epoch)
    model.train()
    optimizer.zero_grad()
    tgt_out = make_one_hot(lbl[1:],category)
    tgt_out[tgt_out==0] = 1e-3
    tgt_out[tgt_out==1] = 1-(category-1)*1e-3
    tgt_out = tgt_out.permute((1,2,0)).contiguous()
    out = model(src,tgt,tgt_mask=tgt_mask)
    #outseq should have (batch,class,seq)
    outseq = out.permute(1,2,0).contiguous()
    running_loss = criterion(outseq,tgt_out)
    running_loss.backward()
    optimizer.step()
    print("running loss: {0}".format(running_loss))
    loss_sum.append(running_loss)
model.eval()
src,tgt,lbl = gen_data(category,seqL,bsz,d_model,100.)
memory = model.encode(src,None,None).detach()
out,atw = model.greedy_decode(memory,2,len(tgt)+1)
true = lbl[1:-1,:]
pred = out[1:-1,:] #note greedy_decode has taken care of argmax 
a = torch.as_tensor(torch.sum(true==pred).detach(),dtype=torch.float)
print("Lbl seq: {0}".format(true))
print("Leaned seq: {0}".format(pred))
print("Accuracy: {0:2f}".format(a/true.nelement()))
#fig,axs = plt.subplots(1,1,figsize=(10,10))
#seaborn.heatmap(np.asarray(atw.squeeze(0)),square=True,vmin=0.0,vmax=1.0,cbar=False,ax=axs)
#plt.show()
#print("Learned seq: {0}".format(torch.exp(out.detach()).argmax(dim=2)))
plt.plot(loss_sum)
plt.xlabel("epoch")
plt.ylabel("KLDiv Loss")
plt.title("KLDiv Loss per epoch for a sequence of length {0}\n(lr={1}, momentum={2})".format(seqL,lr,mom))
plt.show()

#TODO: check if bos/eos caused problem

#TODO: check transformer code
#Perhaps use an embedding layer

#model = FFN(d_model,category+2)
#For feedforward network to do this simple task
#for epoch in range(nepoch):
#    print(epoch)
#    model.train()
#    optimizer.zero_grad()
#    out = model(src) 
#    outseq = out.permute((0,2,1)).contiguous()
#    running_loss = criterion(outseq,lbl)
#    running_loss.backward()
#    optimizer.step()
#    print("running loss: {0}".format(running_loss))
#    loss_sum += running_loss.item()
#model.eval()
#out = model(src)
#sftmx = nn.Softmax(dim=2)
#
#class FFN(nn.Module):
#    def __init__(self,d_in,d_out):
#        super(FFN,self).__init__()
#        self.Linear1 = nn.Linear(d_in,128)
#        self.Linear2 = nn.Linear(128,d_out)
#
#    def forward(self,x):
#        return self.Linear2(F.relu(self.Linear1(x)))
#
#print(sftmx(out))
#print("Lbl seq: {0}".format(lbl))
#print("Learned seq: {0}".format(sftmx(out).argmax(dim=2)))


