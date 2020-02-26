#!/usr/bin/env python3
#general purpose library for funcitons that I'd like to use

#Misc
def pr(x):
    print(x)
    input()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_recolist(filename):
    recos = []
    with open(filename,'r') as f:
        for line in f:
            recos.append(line.strip())
    return recos

def normalize_Data(data,mean):
    #subtract global mean
    print("Data Normalization ...")
    data =  np.subtract(data,mean)
    #length normalization to sqrt(feat-dim)
    feat_dim = np.shape(data)[1]
    l2_vect= np.sqrt(np.sum(np.square(data),axis=1,keepdims=True))
    data = np.divide(np.sqrt(feat_dim)*data,l2_vect)
    return data

#ML
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

def cal_diff(yy,y):
    #yy is prediction, y is truth
    assert(len(yy)==len(y))
    mask = y==1 #positive class
    t_pos = torch.sum(yy[mask]==y[mask]).numpy() #num of true positives
    t_neg = torch.sum(yy[~mask]==y[~mask]).numpy() #num of true negative
    f_pos = torch.sum(yy[~mask]!=y[~mask]).numpy() #num of false positive
    f_neg = torch.sum(yy[mask]!=y[mask]).numpy() #num of false negative
    return t_pos,t_neg,f_pos,f_neg

