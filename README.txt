main steps:

step 0) ami_pre_dnn.sh












breakdowns: 

step 0 breakdown:
0) ami_prep_data.sh  -- prepare ami data
1) ami_extract_mfcc.sh -- extract mfcc of ami
2) ami_run_gen_lbl.sh -- get non-overlapping segments per recording from rttm and pass to downstream, subsequently call ami_gen_label.py
          |
    ami_gen_label.py -- extract vad label from supported segments
3) ami_cat_data.sh -- provide all mfcc/lbl files in list, subsequently call ami_cat_data.py 
          |
    ami_cat_data.py -- combine file presented in file lists of input 


