import user_features as uf
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from transformers import pipeline
from transformers import T5Tokenizer, T5EncoderModel
from model import MLP, MLP_text
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import tqdm
import argparse
def get_env_var(varname, default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        print(varname, ':', var)
    else:
        var = default
        print(varname, ':', var, '(Default)')
    return var

def detect(user, complete=False):
    #TODO load tweets
    timeline = []

    # if len(timeline)==0:
    #     user = api.get_user(screen_name=username)
    #     single = True
    # else:
    #     user = timeline[0]
    online = True
    single = False

    upper, lower = uf.upper_lower_username_cnt(user, online, single)
    entropy = uf.ShannonEntropyAndNomalize(user, online, single)


    single_feature = [
        uf.status_cnt(user, online, single),
        uf.followers_cnt(user, online, single),
        uf.following_cnt(user, online, single),
        uf.listed_cnt(user, online, single),
        uf.default_profile_image(user, online, single),
        uf.is_verified(user, online, single),
        uf.get_user_id(user, online, single),
        uf.has_description(user, online, single),
        uf.protected(user, online, single),
        uf.get_digits_screen_name(user, online, single),
        uf.get_digits_username(user, online, single),
        uf.has_location(user, online, single),
        uf.num_of_hashtags(user, online, single),
        uf.num_of_URLs(user, online, single),
        uf.has_url(user, online, single),
        uf.user_age(user, 'Twibot-22', online, single),
        uf.has_bot_word_in_description(user, online, single),
        uf.has_bot_word_in_screen_name(user, online, single),
        uf.has_bot_word_in_username(user, online, single),
        uf.get_screen_name_length(user, online, single),
        uf.get_username_length(user, online, single),
        uf.get_description_length(user, online, single),
        upper,
        lower,
        uf.get_followers_followees(user, online, single),
        uf.get_number_count_in_screen_name(user, online, single),
        uf.get_number_count_in_username(user, online, single),
        uf.hashtags_count_in_username(user, online, single),
        uf.hashtags_count_in_description(user, online, single),
        uf.urls_count_in_description(user, online, single),
        uf.def_image(user, online, single),
        uf.def_profile(user, online, single),
        uf.tweet_freq(user, online, single),
        uf.followers_growth_rate(user, online, single),
        uf.friends_growth_rate(user, online, single),
        uf.screen_name_unicode_group(user, online, single),
        uf.des_sentiment_score(user, online, single),
        uf.lev_distance_username_screen_name(user, online, single),
        entropy
    ]
    # print(len(single_feature))
    if not complete:
        single_feature = np.array(single_feature).reshape(1, -1)
        RF_pred = RF.predict_proba(single_feature)
        return RF_pred[0][1]
    else:
        single_feature = np.array(single_feature).reshape(1, -1)
        t5_des = uf.get_des_embedding(user, t5_extract, 512, single)
        t5_tweet = uf.get_tweets_embedding(timeline, t5_extract, 512)
        roberta_des = uf.get_des_embedding(user, roberta_extract, 768, single)
        roberta_tweet = uf.get_tweets_embedding(timeline, roberta_extract, 768)

        num = uf.get_num_feat(user, single)
        cat = uf.get_cat_feat(user, single)

        RF_pred = RF.predict_proba(single_feature)
        Ada_pred = Adaboost.predict_proba(single_feature)

        hgt_pred = mlp_hgt(num, cat, roberta_tweet, roberta_des)
        simplehgn_pred = mlp_simplehgn(num, cat, roberta_tweet, roberta_des)
        rgcn_pred = mlp_rgcn(num, cat, roberta_tweet, roberta_des)
        rgt_pred = mlp_rgt(num, cat, roberta_tweet, roberta_des)

        roberta_pred = mlp_roberta(roberta_tweet, roberta_des)
        t5_pred = mlp_t5(t5_tweet, t5_des)

        # calibration
        hgt_pred = torch.softmax(hgt_pred / 1.828, dim=-1)  #
        simplehgn_pred = torch.softmax(simplehgn_pred / 1.818, dim=-1)  #
        rgt_pred = torch.softmax(rgt_pred / 1.826, dim=-1)  #
        rgcn_pred = torch.softmax(rgcn_pred / 1.827, dim=-1)  #

        roberta_pred = torch.softmax(roberta_pred / 1.560, dim=-1)  #
        t5_pred = torch.softmax(t5_pred / 1.552, dim=-1)

        RF_pred = torch.softmax(torch.from_numpy(RF_pred) / 1.415, dim=-1)
        Ada_pred = torch.softmax(torch.from_numpy(Ada_pred) / 1.498, dim=-1)

        pred_stack = torch.stack([
            RF_pred,
            Ada_pred,
            hgt_pred.detach().cpu(),
            simplehgn_pred.detach().cpu(),
            rgt_pred.detach().cpu(),
            rgcn_pred.detach().cpu(),
            roberta_pred.detach().cpu(),
            t5_pred.detach().cpu()
        ], dim=-1)

        weight = torch.tensor([[1.0990, 1.0991, 0.9009, 0.9008, 0.9008, 0.9008, 0.8996, 0.9006]]).t().double()
        pred_all = torch.matmul(pred_stack, weight).squeeze(-1)
        pred_binary = torch.argmax(pred_all, dim=1)

        pred_all = pred_all / pred_all.sum()
        return pred_all[:, 1].item()
        # print('-' * 100)
        # print("pred_score: \t human: {}, bot: {}".format(pred_all[:, 0].item(), pred_all[:, 1].item()))
        # if pred_binary == 0:
        #     print('Human!')
        # else:
        #     print("Bot!")
        # print('-' * 100)

if __name__ == '__main__':
    user_set = 'just_another_day'
    complete=True
    path_data = os.path.join('/scratch/mt4493/bot_detection/data/user_profiles', user_set)
    input_files_list = list(Path(path_data).glob('*.parquet'))
    # define env var
    SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 0)
    SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
    SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
    # define path list
    parquet_path_list = list(np.array_split(
        input_files_list,
        SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
    print(f'# parquets files allocated: {len(parquet_path_list)}')
    # load model
    with open("./checkpoint/RandomForest.pkl", 'rb') as f:
        RF = pickle.load(f)
    print('loaded RF')
    if complete:
        with open("./checkpoint/Adaboost.pkl", 'rb') as f:
            Adaboost = pickle.load(f)

        mlp_hgt = MLP(5, 3, 768, 768, 1024, 0.3)
        hgt_state = torch.load('./checkpoint/HGT.pt', map_location='cpu')
        mlp_hgt.load_state_dict(hgt_state)

        mlp_simplehgn = MLP(5, 3, 768, 768, 1024, 0.3)
        simplehgn_state = torch.load('./checkpoint/SimpleHGN.pt', map_location='cpu')
        mlp_simplehgn.load_state_dict(simplehgn_state)

        mlp_rgt = MLP(5, 3, 768, 768, 1024, 0.3)
        RGT_state = torch.load('./checkpoint/RGT.pt', map_location='cpu')
        mlp_rgt.load_state_dict(RGT_state)

        mlp_rgcn = MLP(5, 3, 768, 768, 1024, 0.3)
        rgcn_state = torch.load('./checkpoint/RGCN.pt', map_location='cpu')
        mlp_rgcn.load_state_dict(rgcn_state)

        mlp_roberta = MLP_text(768, 768, 128, 0.3)
        roberta_state = torch.load('./checkpoint/text-RoBERTa.pt', map_location='cpu')
        mlp_roberta.load_state_dict(roberta_state)

        mlp_t5 = MLP_text(512, 512, 128, 0.3)
        t5_state = torch.load('./checkpoint/text-T5.pt', map_location='cpu')
        mlp_t5.load_state_dict(t5_state)

        mlp_hgt.eval()
        mlp_simplehgn.eval()
        mlp_rgt.eval()
        mlp_rgcn.eval()
        mlp_t5.eval()
        mlp_roberta.eval()

        pred_all = []
        roberta_extract = pipeline('feature-extraction',
                                   model='roberta-base',
                                   tokenizer='roberta-base',
                                   device='cpu',
                                   padding=True,
                                   truncation=True,
                                   max_length=50,
                                   add_special_tokens=True)

        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5EncoderModel.from_pretrained('t5-small')
        t5_extract = pipeline('feature-extraction',
                              model=model,
                              tokenizer=tokenizer,
                              device='cpu',
                              padding=True,
                              truncation=True,
                              max_length=50,
                              add_special_tokens=True)
    output_path = '/scratch/mt4493/bot_detection/results'
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    if complete:
        output_user_set = f'{user_set}_complete'
    else:
        output_user_set = user_set
    output_folder = os.path.join(output_path, output_user_set)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    for path in parquet_path_list:
        print(f'loading {path}')
        df = pd.read_parquet(path)
        print(f'loaded data. shape: {df.shape[0]}')
        bot_pred = list()
        for i, user in enumerate(df.to_dict('records')):
            if i % 100 == 0:
                print(i)
            try:
                pred = detect(user, complete=complete)
                bot_pred.append(pred)
            except Exception as e:
                print(e)
                bot_pred.append(None)
        df['bot_pred'] = bot_pred
        df.to_parquet(os.path.join(output_folder, path.name), index=False)
