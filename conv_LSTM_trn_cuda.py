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
    parser.add_argument("--lstmlyr", type=int, default=1)
    parser.add_argument("--bidirect", type=str2bool, default=False)
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
    def __init__(self,lr,niter,lstmlyr,bidirect):
        self.din = 13 #TODO, change hyperparam when needed
        self.dout = 2
        self.dhid = 12
        self.lstmlyr = lstmlyr
        self.bidirect = bidirect
        self.lr = 0.001
        self.bsize = 10
        self.niter = niter
        self.bidi = "bi" if bidirect else ""

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
    hp = hyperparam(args.lr,args.niter,args.lstmlyr,args.bidirect)
    #preprocess, datasets and dataloader
    data = main_init()
    datasets = data_load_datasets(data,global_mean)
    #configure nn models
#   Conv-LSTM
    trnldr,devldr,tstldr = data_define_dataloader(datasets,hp.bsize,pad_collate_conv)
    model_name = "run{0}_ly{1}_ep{2}_lr{3}_conv{4}lstm".format(args.exp,
                                hp.lstmlyr,hp.niter,hp.lr,hp.bidi)
    lstm = ConvLSTM(hp.din,hp.dout,hp.lstmlyr,hp.bidirect).to(device=device)

#   LSTM
#    trnldr,devldr,tstldr = data_define_dataloader(datasets,hp.bsize,pad_collate)
#    model_name = "run0_ly{0}_ep{1}_lr{2}_h{3}_{4}lstm".format(hp.lstmlyr,hp.niter,hp.lr,hp.dhid,hp.bidi)
#    lstm = LSTMNet(hp.din,hp.dout,hp.dhid,hp.lstmlyr,hp.bidirect).to(device=device)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.SGD(lstm.parameters(),lr=hp.lr,momentum=0.9)
    #load previous trained progress
    if args.model:
        lstm.load_state_dict(torch.load("results/rnn_params/tmp/{0}".format(args.model), map_location=device))
        #lstm.load_state_dict(torch.load("results/rnn_params/{0}".format(model_name), map_location=device))
    #Train and test
    print(model_name)
    lstm = model_train_seq(device,hp.niter,optimizer,criterion,lstm,
                           (trnldr,devldr),name=model_name)
    model_eval_seq(device,lstm,tstldr,criterion,mode="Evaluation",s=model_name)

if __name__=="__main__":
    main()


