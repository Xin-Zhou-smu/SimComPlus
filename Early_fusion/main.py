import argparse
from padding import padding_data
import pickle
import numpy as np 
from evaluation import evaluation_model
from train import train_model
from utils import read_hand_features, preprocess_hand_features, read_hand_features2
import time
import torch
import random
import os
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer


def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	#torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True





def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', action='store_true', help='training DeepJIT model')
    parser.add_argument('-do_valid', action='store_true', help='validing DeepJIT model')
    parser.add_argument('-predict', action='store_true', help='predicting testing data')
    parser.add_argument('-load_model', type=str, help='loading our model')
    parser.add_argument('-msg_length', type=int, default=256, help='the length of the commit message')
    parser.add_argument('-code_line', type=int, default=10, help='the number of LOC in each hunk of commit code')
    parser.add_argument('-code_length', type=int, default=512, help='the length of each LOC of commit code')
    parser.add_argument('-embedding_dim', type=int, default=64, help='the dimension of embedding vector')
    parser.add_argument('-filter_sizes', type=str, default='1, 2, 3', help='the filter size of convolutional layers')
    parser.add_argument('-num_filters', type=int, default=64, help='the number of filters')
    parser.add_argument('-hidden_units', type=int, default=512, help='the number of nodes in hidden layers')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='dropout for training DeepJIT')
    parser.add_argument('-l2_reg_lambda', type=float, default=5e-5, help='regularization rate')
    parser.add_argument('-learning_rate', type=float, default=5e-5, help='learning rate')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=40, help='the number of epochs')
    parser.add_argument('-save-dir', type=str, default='model', help='where to save the snapshot')
    parser.add_argument('-project', type=str, default='openstack')
    parser.add_argument('-combiner', type=str, default='gating_on_cat_and_num_feats_then_sum')
    parser.add_argument('-num_normalization', type=str, default='quantile_normal')
    parser.add_argument('-cat_normalization', type=str, default='ohe')
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the GPU')
    return parser

if __name__ == '__main__':
    params = read_args().parse_args()

    seed_torch(params.seed)
    if params.project == 'openstack':
        params.l2_reg_lambda =0.0001
    elif params.project == 'qt':
        params.l2_reg_lambda =8e-05
    elif params.project == 'jdt':
        params.l2_reg_lambda =2e-05
    elif params.project == 'gerrit':
        params.l2_reg_lambda =5e-05
    elif params.project == 'go':
        params.l2_reg_lambda =8e-05
    elif params.project == 'platform':
        params.l2_reg_lambda =0.0002

    

    params.dictionary_data = '../data/commit_cotents/processed_data/' + params.project + '/' +  params.project  +'_dict.pkl'
    params.train_data =  '../data/commit_cotents/processed_data/' + params.project + '/val_train/' + params.project + '_train.pkl'
    params.val_data = '../data/commit_cotents/processed_data/'  + params.project + '/val_train/' + params.project + '_val.pkl'
    params.pred_data = '../data/commit_cotents/processed_data/' + params.project + '/' + params.project + '_test.pkl'

    params.save_dir = params.save_dir + '/' + params.project + '/' + params.combiner+'_' + params.num_normalization + '_' + params.cat_normalization +'_lr' + str(params.l2_reg_lambda) +'_seed' + str(params.seed)

    data = pickle.load(open(params.train_data, 'rb'))
    ids, labels, msgs, codes = data
    df, test_df = read_hand_features2(params.project)

    train_df = df[:len(ids)]
    val_df = df[len(ids):]
    cat_cols = ['ns', 'nf', 'fix', 'nd']
    numerical_cols = ['entrophy', 'age', 'rexp', 'la', 'ld', 'lt', 'nuc', 'exp', 'sexp']
    train_hand, val_hand, test_hand = preprocess_hand_features(categorical_cols=cat_cols, numerical_cols=numerical_cols,
                                                               train_df=train_df, val_df=val_df, test_df=test_df,
                                                               numerical_transformer_method=params.num_normalization,
                                                               categorical_encode_type=params.cat_normalization, )

    if params.train is True:
        data = pickle.load(open(params.train_data, 'rb'))
        ids, labels, msgs, codes = data 
        labels = np.array(labels)
        val_data = pickle.load(open(params.val_data, 'rb'))
        v_ids, v_labels, v_msgs, v_codes = val_data
        v_labels = np.array(v_labels)
        dictionary = pickle.load(open(params.dictionary_data, 'rb'))   
        dict_msg, dict_code = dictionary


        def noisy_label(real_labels, noisy_rate=0):
            noisy_len = int(len(real_labels)*noisy_rate)
            noisy_labels = []
            for i in range(len(real_labels)):

                if i < noisy_len:
                    if real_labels[i] == 1:
                        noisy_labels.append(0)
                    else:
                        noisy_labels.append(1)

                else:
                    noisy_labels.append(real_labels[i])

            return np.array(noisy_labels)


        labels = noisy_label(labels, noisy_rate=0.0)
        v_labels = noisy_label(v_labels, noisy_rate=0.0)









        pad_msg = padding_data(data=msgs, dictionary=dict_msg, params=params, type='msg')        
        pad_code = padding_data(data=codes, dictionary=dict_code, params=params, type='code')

        v_pad_msg = padding_data(data=v_msgs, dictionary=dict_msg, params=params, type='msg')
        v_pad_code = padding_data(data=v_codes, dictionary=dict_code, params=params, type='code')

        data = (pad_msg, pad_code, labels, dict_msg, dict_code, train_hand[0], train_hand[1])

        v_data = (v_pad_msg, v_pad_code, v_labels, dict_msg, dict_code, val_hand[0], val_hand[1])

        train_model(data=data, val=v_data, params=params)

    
    elif params.predict is True:
        

        params.load_model = params.save_dir + '/best_model.pt'


        data = pickle.load(open(params.pred_data, 'rb'))
        ids, labels, msgs, codes = data 
        labels = np.array(labels)        

        dictionary = pickle.load(open(params.dictionary_data, 'rb'))   
        dict_msg, dict_code = dictionary



        pad_msg = padding_data(data=msgs, dictionary=dict_msg, params=params, type='msg')        
        pad_code = padding_data(data=codes, dictionary=dict_code, params=params, type='code')
        
        starttime = time.time()
        data = (pad_msg, pad_code, labels, dict_msg, dict_code, test_hand[0], test_hand[1])
        evaluation_model(data=data, params=params)
        endtime = time.time()
        pred_time = endtime - starttime 
        print('predicting time needed:', pred_time)
        

    else:
        print('--------------------------------------------------------------------------------')
        print('--------------------------Something wrongs with your command--------------------')
        print('--------------------------------------------------------------------------------')
        exit()


