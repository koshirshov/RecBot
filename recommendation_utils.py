import pandas as pd

def create_recommendations_from_one_model(model, list_users):
    df_recommendations = pd.DataFrame({'user_id': list_users})
    df_recommendations['item_id'] = model.recommend(users = list_users)
    df_recommendations = df_recommendations.explode('item_id')
    df_recommendations['rank'] = df_recommendations.groupby('user_id').cumcount() + 1
    return df_recommendations

def create_submission_file(df_recommendations):
    df_recommendations = df_recommendations.groupby('user_id').agg({'item_id': list}).reset_index()
    df_recommendations.to_csv('/home/koshirshov/koshirshov/RecBot/data/submissions/' + 'sample_submission.csv', index=False)
    
#from collections import Counter
#genres = Counter()
#for genres_ in df['genres'].str.split(','):
#    for genre in genres_:
#        genres[genre.strip()] += 1
#
#popular_genres = set()
#for genre in genres.most_common(config.num_genres):
#    popular_genres.add(genre[0])
#    
##df['genres'] = df['genres'].apply(lambda x: process_genre(x, popular_genres))

#def process_genre(list_genres, popular_genres):
#    x = sorted([genre.strip() for genre in list_genres.split(',') if genre.strip() in popular_genres])
#    return ', '.join(x)

#def generate_implicit_recs_mapper(
#    model,
#    train_matrix,
#    top_N,
#    user_mapping,
#    item_inv_mapping,
#    filter_already_liked_items
#):
#    def _recs_mapper(user):
#        user_id = user_mapping[user]
#        recs = model.recommend(user_id, 
#                               train_matrix, 
#                               N=top_N, 
#                               filter_already_liked_items=filter_already_liked_items)
#        return [item_inv_mapping[item] for item, _ in recs]
#    return _recs_mapper
#
#def get_coo_matrix(df, 
#                   user_col='user_id', 
#                   item_col='item_id', 
#                   weight_col=None, 
#                   users_mapping={}, 
#                   items_mapping={}):
#    
#    if weight_col is None:
#        weights = np.ones(len(df), dtype=np.float32)
#    else:
#        weights = df[weight_col].astype(np.float32)
#
#    interaction_matrix = sp.coo_matrix((
#        weights, 
#        (
#            df[user_col].map(users_mapping.get), 
#            df[item_col].map(items_mapping.get)
#        )
#    ))
#    return interaction_matrix