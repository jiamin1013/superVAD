#! /usr/bin/env python3
# Copyright  2019  Jiamin Xie
#
# --- imports ---
import sys
import pdb
import time
import random
import argparse
import torch.optim as optim
import kaldi_io as kio
import matplotlib.pyplot as plt
from xl_encoder_dataset import *
from xl_encoder_model import *
from xl_encoder_train_eval import *
from xl_encoder_misc import *

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
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--nlayer", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout",type=float,default=0.1)
    parser.add_argument("--disable_cuda",type=str2bool,default=True,
                        help = """disable using GPU""")
    parser.add_argument("--model",type=str,default=None)
    parser.add_argument("--exp",type=str,default="0") #experiment number

    args = parser.parse_args()
    return args

def main_init():
    #initialization of main, usually instantiate required files
    trn_utts = read_recolist("pyprep/train/reco_list_sorted")
    dev_utts = read_recolist("pyprep/dev/reco_list_sorted")
    tst_utts = read_recolist("pyprep/test/reco_list_sorted")
    return trn_utts,dev_utts,tst_utts

class hyperparam():
    def __init__(self,lr,niter,nhead,nlayer,d_ff,dropout):
        self.d_model = 128
        self.dout = 2 #ignore eos
        self.dhid = d_ff
        self.nhead = nhead
        self.nlayer = nlayer
        self.dropout = dropout
        self.lr = lr
        self.bsize = 1#2#10
        self.niter = niter

def main():
    #set seeds
    np.random.seed(0)
    torch.manual_seed(0)
    args = get_args()
    print(args.disable_cuda)
    print(torch.cuda.is_available())
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    #setup hyperparameters 
    hp = hyperparam(args.lr,args.niter,args.nhead,args.nlayer,args.d_ff,args.dropout)
    #global mean
    global_mean = np.mean(np.load("pyprep/train/train_all_mfcc.npy"),
                          axis=0,keepdims=True) if args.norm else 0
    #preprocess, datasets and dataloader
    data = main_init()
    datasets = data_load_datasets(data,global_mean)
    trnldr,devldr,tstldr = data_define_dataloader(datasets,hp.bsize,pad_collate)
    model_name = "run{0}_nhead{1}_ly{2}_ep{3}_lr{4}_convxl".format(args.exp,hp.nhead,hp.nlayer,hp.niter,hp.lr)
    #Make Transformer Model
    model = make_conv_encoder_model(13,128,2,hp.nhead,hp.nlayer,512,hp.dropout)
    #define loss function
    loss_fn = LabelSmoothing(size=2,smoothing=10e-5,pad_val=-100)
    optimizer = optim.SGD(model.parameters(),lr=hp.lr,momentum=0.9)
    #load previous trained progress
    if args.model:
        model.load_state_dict(torch.load("results/rnn_params/{0}".format(args.model), map_location=device))
        print("Pretrained model {} loaded ".format(args.model))
    else:
        print(model_name)
    #Train and test
    #TODO: write function to train and evaluate tx model
    model,losses = model_train(device,hp.niter,optimizer,loss_fn,model,(trnldr,devldr),name=model_name)
    model_eval(model,tstldr,loss_fn,mode="Evaluation")
#    np.save("loss_4x4.npy",losses)
    
if __name__=="__main__":
    main()


