import config
from datetime import timedelta
from itertools import islice, cycle
from pandas_utils import save_data
from scipy.sparse import csr_matrix
from lightfm import LightFM
from sklearn.preprocessing import normalize
from tqdm import tqdm
import numpy as np


class PopularRecommender():
    def __init__(self, days=30, n=config.num_recommendations, item_column='item_id', dt_column='date'):
        self.n = n
        self.days = days
        self.item_column = item_column
        self.dt_column = dt_column
        
    def fit(self, train_interactions):
        min_date = train_interactions[self.dt_column] - timedelta(days = self.days)
        self.recommendations = train_interactions.loc[train_interactions[self.dt_column] > min_date, self.item_column].value_counts().head(self.n).index.values
        save_data(config.models_data_path + 'model_popular.pkl', self)

    def recommend(self, users=None):
        recommendations = self.recommendations[:self.n]
        if users is None:
            return recommendations
        else:
            return list(islice(cycle([recommendations]), len(users)))


class LightFmModel():
    def __init__(self, n=config.num_recommendations, user_column='user_id', item_column='item_id'):
        self.n = n
        self.item_column = item_column
        self.user_column = user_column
        self.model = LightFM(
            loss="warp",
            learning_rate=config.learning_rate,
            random_state=42,
            no_components=config.no_component,
            item_alpha=config.item_alpha,
            user_alpha=config.user_alpha,
            max_sampled=config.max_sampled,
        )

    def prepare_csr_matrix(self, train_interactions, quantity_column = 'quantity'):
        train_interactions[quantity_column] = 1
        all_users = set(train_interactions[self.user_column])
        #train_interactions = train_interactions[train_interactions['watched_pct'] > 25]
        good_users = train_interactions.groupby(self.user_column)[quantity_column].sum()
        good_users = set(good_users[good_users >= config.min_interactions_user].index)
        good_items = train_interactions.groupby(self.item_column)[quantity_column].sum()
        good_items = set(good_items[good_items >= config.min_interactions_item].index)

        train_interactions = train_interactions[train_interactions[self.user_column].isin(good_users)]
        train_interactions = train_interactions[train_interactions[self.item_column].isin(good_items)]
        self.users_not_in_matrix = all_users - set(train_interactions[self.user_column])

        good_users = set(train_interactions[self.user_column])
        good_items = set(train_interactions[self.item_column])
        self.users_mapping = dict(zip(good_users, range(len(good_users))))
        self.items_mapping = dict(zip(good_items, range(len(good_users))))
        self.users_mapping_reverse = {value:key for (key,value) in self.users_mapping.items()}
        self.items_mapping_reverse = {value:key for (key,value) in self.items_mapping.items()}

        train_interactions.loc[:, "le_user_id"] = train_interactions.loc[:, self.user_column].map(self.users_mapping)
        train_interactions.loc[:, "le_item_id"] = train_interactions.loc[:, self.item_column].map(self.items_mapping)
        train_interactions = train_interactions.groupby(["le_user_id", "le_item_id"])[quantity_column].sum().reset_index()

        self.train_matrix = csr_matrix(
        (
            train_interactions[quantity_column],
            (
                train_interactions["le_user_id"],
                train_interactions["le_item_id"],
            ),
        ),
        shape=(len(good_users), len(good_items)),
        )

    def fit(self, num_epochs):
        """fit lightfm"""
        for _ in tqdm(range(num_epochs), desc="fit light_fm model, epochs:"):
            self.model.fit_partial(
                self.train_matrix,
                epochs=1,
                verbose=False,
                num_threads=config.num_threads,
            )
        save_data(config.models_data_path + 'model_lfm.pkl', self)

    def recommend(self, users, normalize_embs = False):
        """batch predict from lightfm"""
        
        mapper = lambda t: self.users_mapping[t]
        mapper_func = np.vectorize(mapper)
        users = mapper_func(users)
        
        users_mapping_predict = {user_id: i for i, user_id in enumerate(users)}
        
        if normalize_embs:
            self.model.item_biases = np.zeros_like(self.model.item_biases)
            self.model.user_biases = np.zeros_like(self.model.user_biases)
            self.model.item_embeddings = normalize(self.model.item_embeddings)
            self.model.user_embeddings = normalize(self.model.user_embeddings)

        user_batch_size = config.user_batch_size
        user_count = len(users)
        item_count = self.train_matrix.shape[1]

        predictions_array = np.empty(shape=(user_count, self.n), dtype=int)
        predictions_array[:, :] = 0

        for start in tqdm(
            range(0, user_count, user_batch_size),
            desc="predict from light_fm model, users:",
        ):
            end = start + user_batch_size
            if end > user_count:
                end = user_count
            user_ids = np.repeat(np.array(users[start:end]), item_count)
            item_ids = np.tile(np.arange(item_count), (end - start))
            pred = self.model.predict(
                user_ids=user_ids,
                item_ids=item_ids,
                num_threads=config.num_threads,
            ).reshape((end - start), item_count)
            scores = np.partition(pred, -np.arange(self.n))
            pred = np.argpartition(pred, -np.arange(self.n))
            predictions_array[start:end] = np.flip(
                pred[:, -self.n :], axis=1
            )

        mapper = lambda t: self.items_mapping_reverse[t]
        mapper_func = np.vectorize(mapper)
        predictions_array = mapper_func(predictions_array)

        predictions = list()
        for _, user_id in enumerate(users):
            predictions.append(predictions_array[users_mapping_predict[user_id]])

        return predictions