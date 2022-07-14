from datetime import date
date_test = date(2021, 8, 16)
initial_data_path = '/home/koshirshov/koshirshov/RecBot/data/initial/'
proceed_data_path = '/home/koshirshov/koshirshov/RecBot/data/proceed/'
models_data_path = '/home/koshirshov/koshirshov/RecBot/data/models/'
num_recommendations = 10
#light_fm_model:
min_interactions_user = 5
min_interactions_item = 10
learning_rate = 0.005
no_component = 100
user_alpha = 0.00001
item_alpha = 0.00001
max_sampled = 20
user_batch_size = 10000
num_threads = 8