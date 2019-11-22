#! /usr/bin/env python3
# Copyright  2019  Jiamin Xie
#
# --- imports ---
import sys
import pdb
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import kaldi_io as kio
from mylib import *
from nnlib import *

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
    parser.add_argument("--niter", type=int, default=100,
                        help="""epoch number""")
    parser.add_argument("--disable_cuda",type=str2bool,default=True,
                        help = """disable using GPU""")
    args = parser.parse_args()
    return args

def main_init():
    #initialization of main, usually instantiate required files
    trn_utts = read_recolist("pyprep/train/reco_list_sorted")
    dev_utts = read_recolist("pyprep/dev/reco_list_sorted")
    tst_utts = read_recolist("pyprep/test/reco_list_sorted")
    return trn_utts,dev_utts,tst_utts

class hyperparam():
    def __init__(self,lr,niter):
        self.din = 13
        self.dout = 2
        self.dhid = 12
        self.lstmlyr = 2
        self.lr = lr
        self.bsize = 10
        self.niter = niter

def main():
    #set seeds
    np.random.seed(0)
    torch.manual_seed(0)
    args = get_args()
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    #setup hyperparameters 
    global_mean = np.mean(np.load("pyprep/train/train_all_mfcc.npy"),
                          axis=0,keepdims=True) if args.norm else 0
    hp = hyperparam(args.lr,args.niter)
    model_name = "run3_ly{0}_ep{1}_lr{2}_h{3}_lstm".format(hp.lstmlyr,hp.niter,hp.lr,hp.dhid)
    #preprocess, datasets and dataloader
    data = main_init()
    datasets = data_load_datasets(data,global_mean)
    trnldr,devldr,tstldr = data_define_dataloader(datasets,hp.bsize,pad_collate)
    #configure nn models
    lstm = LSTMNet(hp.din,hp.dout,hp.dhid,hp.lstmlyr).to(device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.SGD(lstm.parameters(),lr=hp.lr,momentum=0.9)
    #load previous trained progress
#    lstm.load_state_dict(torch.load("results/rnn_params/run2_ly2_ep500_lr0.001_h12_lstm"))
    #Test
    lstm = model_train_seq(device,niter,optimizer,criterion,lstm,
                           (trnldr,devldr),name=model_name)
    model_eval_seq(device,lstm,tstldr,criterion,mode="Evaluation")

if __name__=="__main__":
    main()


