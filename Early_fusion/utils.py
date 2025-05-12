import numpy as np
import math
import os, torch
import random
import pandas as pd
import pickle

from multimodal_transformers.data.data_utils import (
    CategoricalFeatures,
    agg_text_columns_func,
    convert_to_func,
    get_matching_cols,
    load_num_feats,
    load_cat_and_num_feats,
    normalize_numerical_feats,
)
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

def read_hand_features(project):


    train_file_ = '../data/hand_crafted_features/' + project + '/k_train.csv'
    df = pd.read_csv(train_file_)
    test_file_ = '../data/hand_crafted_features/' + project + '/k_test.csv'
    df2 = pd.read_csv(test_file_)

    f = open('../data/commit_cotents/processed_data/' + project + '/' + project + '_train.pkl', 'rb')
    content_train = pickle.load(f)
    id_train, label_train, msg_train, cod_train = content_train
    id_postion_train = dict()
    for i in range(len(id_train)):
        id_postion_train[id_train[i]] = i
    f = open('../data/commit_cotents/processed_data/' + project + '/' + project + '_test.pkl', 'rb')
    content_test = pickle.load(f)
    id_test, label_test, msg_test, cod_test = content_test
    id_postion_test = dict()
    for i in range(len(id_test)):
        id_postion_test[id_test[i]] = i
    df2_id = list(df2['_id'])
    df_id = list(df['_id'])
    df_msg, df_code, df2_msg, df2_code = [], [], [], []
    for id_ in df_id:
        df_msg.append(msg_train[id_postion_train[id_]])
        df_code.append(' '.join(cod_train[id_postion_train[id_]]))
    for id_ in df2_id:
        df2_msg.append(msg_test[id_postion_test[id_]])
        df2_code.append(' '.join(cod_test[id_postion_test[id_]]))

    df['msg'] = pd.Series(df_msg)
    df['code'] = pd.Series(df_code)
    df2['msg'] = pd.Series(df2_msg)
    df2['code'] = pd.Series(df2_code)

    return df, df2

def read_hand_features2(project):


    train_file_ = '../data/hand_crafted_features/' + project + '/k_train.csv'
    df = pd.read_csv(train_file_)
    test_file_ = '../data/hand_crafted_features/' + project + '/k_test.csv'
    df2 = pd.read_csv(test_file_)

    f = open('../data/commit_cotents/processed_data/' + project + '/' + project + '_train.pkl', 'rb')
    content_train = pickle.load(f)
    _, label_train, msg_train, cod_train = content_train

    f = open('../data/commit_cotents/processed_data/' + project + '/' + project + '_test.pkl', 'rb')
    content_test = pickle.load(f)
    _, label_test, msg_test, cod_test = content_test


    df['msg'] = pd.Series(msg_train)
    df['code'] = pd.Series(cod_train)
    df2['msg'] = pd.Series(msg_test)
    df2['code'] = pd.Series(cod_test)

    return df, df2


def preprocess_hand_features(categorical_cols, numerical_cols, train_df, val_df, test_df, numerical_transformer_method='quantile_normal', categorical_encode_type = 'ohe'):
        # numerical_transformer_method='quantile_normal'
        # categorical_encode_type = 'ohe'
        # categorical_cols = cat_cols
        # numerical_cols = numerical_cols


        if categorical_encode_type == 'ohe' or categorical_encode_type == 'binary':
            dfs = [df for df in [train_df, val_df, test_df] if df is not None]
            data_df = pd.concat(dfs, axis=0)
            cat_feat_processor = CategoricalFeatures(data_df, categorical_cols, categorical_encode_type)
            vals = cat_feat_processor.fit_transform()
            cat_df = pd.DataFrame(vals, columns=cat_feat_processor.feat_names)

            # data_df = data_df.drop(['ns',  'nf', 'fix', 'nd'], axis=1)
            data_df = data_df.reset_index(drop=True)
            data_df = pd.concat([data_df, cat_df], axis=1)
            categorical_cols = cat_feat_processor.feat_names

            len_train = len(train_df)
            len_val = len(val_df) if val_df is not None else 0

            train_df = data_df.iloc[:len_train]
            if val_df is not None:
                val_df = data_df.iloc[len_train: len_train + len_val]
                len_train = len_train + len_val
            test_df = data_df.iloc[len_train:]

            categorical_encode_type = None
        if numerical_transformer_method != 'none':
            if numerical_transformer_method == 'yeo_johnson':
                numerical_transformer = PowerTransformer(method='yeo-johnson')
            elif numerical_transformer_method == 'box_cox':
                numerical_transformer = PowerTransformer(method='box-cox')
            elif numerical_transformer_method == 'quantile_normal':
                numerical_transformer = QuantileTransformer(output_distribution='normal')
            else:
                raise ValueError(f'preprocessing transformer method '
                                 f'{numerical_transformer_method} not implemented')
            num_feats = load_num_feats(train_df, convert_to_func(numerical_cols))
            numerical_transformer.fit(num_feats)
        else:
            numerical_transformer = None


        categorical_cols_func = convert_to_func(categorical_cols)
        numerical_cols_func = convert_to_func(numerical_cols)

        train_categorical_feats, train_numerical_feats = load_cat_and_num_feats(train_df,
                                                                    categorical_cols_func,
                                                                    numerical_cols_func,
                                                                    categorical_encode_type)
        train_numerical_feats = normalize_numerical_feats(train_numerical_feats, numerical_transformer)

        val_categorical_feats, val_numerical_feats = load_cat_and_num_feats(val_df,
                                                                                categorical_cols_func,
                                                                                numerical_cols_func,
                                                                                categorical_encode_type)
        val_numerical_feats = normalize_numerical_feats(val_numerical_feats, numerical_transformer)

        test_categorical_feats, test_numerical_feats = load_cat_and_num_feats(test_df,
                                                                                categorical_cols_func,
                                                                                numerical_cols_func,
                                                                                categorical_encode_type)
        test_numerical_feats = normalize_numerical_feats(test_numerical_feats, numerical_transformer)

        return (train_categorical_feats, train_numerical_feats), (val_categorical_feats, val_numerical_feats ), (test_categorical_feats, test_numerical_feats)











def save(model, save_dir, save_prefix, epochs):
    if not os.path.isdir(save_dir):       
        os.makedirs(save_dir)
    #save_prefix = os.path.join(save_dir, save_prefix)
    #save_path = '{}_{}.pt'.format(save_prefix, epochs)
    
    
    save_file = 'best_model.pt'
    save_path = os.path.join(save_dir, save_file)
    torch.save(model.state_dict(), save_path)

def mini_batches_test(X_msg, X_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    shuffled_X_msg, shuffled_X_code, shuffled_Y = X_msg, X_code, Y
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size)))

    for k in range(0, num_complete_minibatches):        
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_code = shuffled_X_code[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:        
        mini_batch_X_msg = shuffled_X_msg[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_code = shuffled_X_code[num_complete_minibatches * mini_batch_size: m, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batches_test_with_ids(ids, X_msg, X_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    shuffled_ids, shuffled_X_msg, shuffled_X_code, shuffled_Y = ids, X_msg, X_code, Y
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size)))

    for k in range(0, num_complete_minibatches):
        mini_batch_ids = shuffled_ids[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_code = shuffled_X_code[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_ids, mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_ids = shuffled_ids[num_complete_minibatches * mini_batch_size: m]
        mini_batch_X_msg = shuffled_X_msg[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_code = shuffled_X_code[num_complete_minibatches * mini_batch_size: m, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_ids, mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batches_train(X_msg, X_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_msg, shuffled_X_code, shuffled_Y = X_msg, X_code, Y
    Y = Y.tolist()
    Y_pos = [i for i in range(len(Y)) if Y[i] == 1]
    Y_neg = [i for i in range(len(Y)) if Y[i] == 0]    

    # Step 2: Randomly pick mini_batch_size / 2 from each of positive and negative labels
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) + 1
    for k in range(0, num_complete_minibatches):
        indexes = sorted(
            random.sample(Y_pos, int(mini_batch_size / 2)) + random.sample(Y_neg, int(mini_batch_size / 2)))        
        mini_batch_X_msg, mini_batch_X_code = shuffled_X_msg[indexes], shuffled_X_code[indexes]
        mini_batch_Y = shuffled_Y[indexes]
        mini_batch = (mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batches_train_handfeatures(X_msg, X_code, X_cat_feat, X_num_feat, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_msg, shuffled_X_code, shuffled_X_cat_feat, shuffled_X_num_feat, shuffled_Y = X_msg, X_code, X_cat_feat, X_num_feat, Y
    Y = Y.tolist()
    Y_pos = [i for i in range(len(Y)) if Y[i] == 1]
    Y_neg = [i for i in range(len(Y)) if Y[i] == 0]

    # Step 2: Randomly pick mini_batch_size / 2 from each of positive and negative labels
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) + 1
    for k in range(0, num_complete_minibatches):
        indexes = sorted(
            random.sample(Y_pos, int(mini_batch_size / 2)) + random.sample(Y_neg, int(mini_batch_size / 2)))
        mini_batch_X_msg, mini_batch_X_code, mini_batch_X_cat_feat, mini_batch_X_num_feat = shuffled_X_msg[indexes], shuffled_X_code[indexes], shuffled_X_cat_feat[indexes], shuffled_X_num_feat[indexes]
        mini_batch_Y = shuffled_Y[indexes]
        mini_batch = (mini_batch_X_msg, mini_batch_X_code, mini_batch_X_cat_feat, mini_batch_X_num_feat, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches



def mini_batches_test_handfeatures(X_msg, X_code, X_cat_feat, X_num_feat, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    shuffled_X_msg, shuffled_X_code, shuffled_X_cat_feat, shuffled_X_num_feat,  shuffled_Y = X_msg, X_code, X_cat_feat, X_num_feat, Y
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size)))

    for k in range(0, num_complete_minibatches):
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_code = shuffled_X_code[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        mini_batch_X_cat_feat = shuffled_X_cat_feat[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_num_feat = shuffled_X_num_feat[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]

        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_code, mini_batch_X_cat_feat, mini_batch_X_num_feat, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X_msg = shuffled_X_msg[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_code = shuffled_X_code[num_complete_minibatches * mini_batch_size: m, :, :]
        mini_batch_X_cat_feat = shuffled_X_cat_feat[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_num_feat = shuffled_X_num_feat[num_complete_minibatches * mini_batch_size: m, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_code, mini_batch_X_cat_feat, mini_batch_X_num_feat, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batches_train_non_balance(X_msg, X_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)
    print('mini_batch_size:', mini_batch_size)
    ## Shuffle Training Data
    indexs = list(range(m))
    random.shuffle(indexs)
    shuffled_X_msg, shuffled_X_code, shuffled_Y = X_msg[indexs], X_code[indexs], Y[indexs]
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size)))

    for k in range(0, num_complete_minibatches):
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_code = shuffled_X_code[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches



def get_world_dict(world2id):
    id2world = dict()
    for world in world2id:
        id2world[world2id[world]] = world
    return id2world


def mapping_dict_world(senten_ids, id2world):
    return [id2world[_id] for _id in senten_ids]
