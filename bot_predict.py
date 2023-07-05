import user_features as uf
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import os

def get_env_var(varname, default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        print(varname, ':', var)
    else:
        var = default
        print(varname, ':', var, '(Default)')
    return var

def detect(user):
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

    single_feature = np.array(single_feature).reshape(1,-1)
#     t5_des = uf.get_des_embedding(user, t5_extract, 512, single)
#     t5_tweet = uf.get_tweets_embedding(timeline, t5_extract, 512)
#     roberta_des = uf.get_des_embedding(user, roberta_extract, 768, single)
#     roberta_tweet = uf.get_tweets_embedding(timeline, roberta_extract, 768)

    num = uf.get_num_feat(user, single)
    cat = uf.get_cat_feat(user, single)

    RF_pred = RF.predict_proba(single_feature)
#     Ada_pred = Adaboost.predict_proba(single_feature)
    return RF_pred

if __name__ == '__main__':
    path_data = '/scratch/mt4493/bot_detection/data/user_profiles'
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
    output_path = '/scratch/mt4493/bot_detection/results'
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    output_folder = os.path.join(output_path, 'NG_users')
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
                pred = detect(user)[0][1]
                bot_pred.append(pred)
            except Exception as e:
                print(e)
                bot_pred.append(None)
        df['bot_pred'] = bot_pred
        df.to_parquet(os.path.join(output_folder, path.name), index=False)
