import config
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')
import pandas as pd
from pandas_utils import save_data


class PreprocessingData():
    def __init__(self, ):
        self.initial_data_path = config.initial_data_path
        self.proceed_data_path = config.proceed_data_path

    def process_users(self, df, ):
        """process users dataframe"""
        df['age'] = df['age'].fillna('age_unknown')
        df['age'] = df['age'].astype('category')
        df['income'] = df['income'].fillna('income_unknown')
        df['income'] = df['income'].astype('category')
        df['sex'] = df['sex'].fillna('sex_unknown')
        df.loc[df['sex'] == 'М', 'sex'] = 'M'
        df.loc[df['sex'] == 'Ж', 'sex'] = 'F'
        df['sex'] = df['sex'].astype('category')
        df['kids_flg'] = df['kids_flg'].astype('bool')
        df = df.reset_index(drop = True)
        save_data(self.proceed_data_path + 'users_processed.pkl', df)
        self.users_processed = df

    def process_items(self, df, ):
        """process items dataframe"""
        df['content_type'] = df['content_type'].astype('category')

        df['title'] = df['title'].str.lower()
        df['title_orig'] = df['title_orig'].str.lower()
        df['title'] = df['title'].fillna('None') 
        df['title_orig'] = df['title_orig'].fillna('None')

        df.loc[df['release_year'].isna(), 'release_year'] = 2020.
        df.loc[df['release_year'] < 1960, 'release_year_cat'] = 'inf_1960'
        df.loc[df['release_year'] >= 2020, 'release_year_cat'] = '2020_inf'
        for i in range (1960, 2020, 10):
            df.loc[(df['release_year'] >= i) & (df['release_year'] < i+10), 'release_year_cat'] = f'{i}-{i+10}'
        for i in range (2010, 2020, 2):
            df.loc[(df['release_year'] >= i) & (df['release_year'] < i+2), 'release_year_cat'] = f'{i}-{i+2}'
        df['release_year_cat'] = df['release_year_cat'].astype('category')

        df['genres'] = df['genres'].str.lower()
        df['genres'] = df['genres'].apply(lambda x: ', '.join(sorted(list(set(x.split(', '))))))
        df['genres'] = df['genres'].astype('category')

        df.loc[df.countries.isna(), 'countries'] = 'Россия'
        df['countries'] = df['countries'].str.lower()
        df['countries'] = df['countries'].apply(lambda x: ', '.join(sorted(list(set(x.split(', '))))))
        df['countries'] = df['countries'].astype('category')

        df['for_kids'] = df['for_kids'].fillna(0)
        df['for_kids'] = df['for_kids'].astype('bool')

        df.loc[df.age_rating.isna(), 'age_rating'] = 0
        df['age_rating'] = df['age_rating'].astype('category')

        df['studios'] = df['studios'].fillna('Unknown')
        df['studios'] = df['studios'].str.lower()
        df['studios'] = df['studios'].apply(lambda x: ', '.join(sorted(list(set(x.split(', '))))))
        df['studios'] = df['studios'].astype('category')

        df['directors'] = df['directors'].fillna('Unknown')
        df['directors'] = df['directors'].str.lower()
        df['directors'] = df['directors'].apply(lambda x: ', '.join(sorted(list(set(x.split(', '))))))
        df['directors'] = df['directors'].astype('category')

        df['actors'] = df['actors'].fillna('Unknown')
        df['actors'] = df['actors'].str.lower()
        df['actors'] = df['actors'].apply(lambda x: ', '.join(sorted(list(set(x.split(', '))))))
        df['actors'] = df['actors'].astype('category')

        df['keywords'] = df['keywords'].fillna('Unknown')
        df['keywords'] = df['keywords'].astype('category')

        df['description'] = df['description'].fillna('-')

        # print(df[df.duplicated(subset=['content_type', 'countries', 'title', 'directors', 'actors'], keep=False)].sort_values('title')[['item_id', 'content_type', 'title','title_orig', 'countries', 'for_kids', 'directors', 'studios', 'actors', 'release_year_cat']])
        # we have duplicate films-fix it in interactions
        self.duplicate_items = {12889 : 13787, 3286 : 2386, 15326 : 995}
        df = df.drop(index = df[df['item_id'].isin(self.duplicate_items.keys())].index)
        df = df.reset_index(drop = True)
        item_titles = pd.Series(df['title'].values, index=df['item_id']).to_dict()
        save_data(self.proceed_data_path + 'items_processed.pkl', df)
        save_data(self.proceed_data_path + 'duplicate_items.pkl', self.duplicate_items)
        save_data(self.proceed_data_path + 'item_titles.pkl', item_titles)
        self.items_processed = df

    def process_interactions(self, df, ):
        """process interactions dataframe"""
        df.loc[df['item_id'].isin(self.duplicate_items.keys()), 'item_id'] = df.loc[df['item_id'].isin(self.duplicate_items.keys()), 'item_id'].map(self.duplicate_items)
        df['watched_pct'] = df['watched_pct'].astype(pd.Int8Dtype())
        df['date'] = df['last_watch_dt'].dt.date
        
        df = df.drop('last_watch_dt', axis = 1)
        df = df.reset_index(drop = True)
        test_df = df[df['date'] >= config.date_test]
        train_df = df[df['date'] < config.date_test]
        save_data(self.proceed_data_path + 'interactions_processed.pkl', df)
        save_data(self.proceed_data_path + 'test_interactions.pkl', test_df)
        save_data(self.proceed_data_path + 'train_interactions.pkl', train_df)
        self.interactions_processed = df

    def process_submission(self, df, ):
        """process submission dataframe"""
        users_to_submit = set(df['user_id'])
        users_with_interactions = set(self.interactions_processed['user_id'])
        all_users = set(self.users_processed['user_id'])
        logging.info(f'There is {len(users_to_submit)} users for submit')
        logging.info(f'There is {len(all_users)} total users')
        logging.info(f'There is {len(users_with_interactions)} users with interactions')
        logging.info(f'There is {len(users_to_submit - users_with_interactions)} users for submit without interactions')
        logging.info(f'There is {len(users_to_submit - all_users)} users for submit without information')
        df['has_interaction'] = False
        df.loc[df['user_id'].isin(users_with_interactions), 'has_interaction'] = True
        df['has_information'] = False
        df.loc[df['user_id'].isin(all_users), 'has_information'] = True
        df = df.reset_index(drop = True)
        save_data(self.proceed_data_path + 'sample_submission_processed.pkl', df)

    def preprocess_data(self, ):
        logging.info('load initial dataframes')
        users_df = pd.read_csv(self.initial_data_path + 'users.csv')
        items_df = pd.read_csv(self.initial_data_path + 'items.csv')
        interactions_df = pd.read_csv(self.initial_data_path + 'interactions.csv', parse_dates=['last_watch_dt'])
        sample_submission = pd.read_csv(self.initial_data_path + 'sample_submission.csv')
        logging.info('start process dataframes')
        self.process_users(users_df)
        self.process_items(items_df)
        self.process_interactions(interactions_df)
        self.process_submission(sample_submission)