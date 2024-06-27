import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import SVD

ratings = pd.read_csv('assets/ratings.csv')
movies = pd.read_csv('assets/movies.csv')

print(ratings.head())
print(movies.head())

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.25)

algo = KNNBasic(sim_options={'user_based': True})
algo.fit(trainset)

predictions = algo.test(testset)
accuracy.rmse(predictions)

def get_user_recommendations(user_id, top_n=10):
    user_inner_id = algo.trainset.to_inner_uid(user_id)
    user_ratings = algo.trainset.ur[user_inner_id]
    recommendations = []
    for iid in algo.trainset.all_items():
        if iid not in [item[0] for item in user_ratings]:
            est_rating = algo.predict(user_id, algo.trainset.to_raw_iid(iid)).est
            recommendations.append((iid, est_rating))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return [algo.trainset.to_raw_iid(iid) for iid, _ in recommendations[:top_n]]

print(get_user_recommendations(1))

algo_item = KNNBasic(sim_options={'user_based': False})
algo_item.fit(trainset)

predictions_item = algo_item.test(testset)
accuracy.rmse(predictions_item)

def get_item_recommendations(user_id, top_n=10):
    user_inner_id = algo_item.trainset.to_inner_uid(user_id)
    user_ratings = algo_item.trainset.ur[user_inner_id]
    recommendations = []
    for iid in algo_item.trainset.all_items():
        if iid not in [item[0] for item in user_ratings]:
            est_rating = algo_item.predict(user_id, algo_item.trainset.to_raw_iid(iid)).est
            recommendations.append((iid, est_rating))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return [algo_item.trainset.to_raw_iid(iid) for iid, _ in recommendations[:top_n]]

print(get_item_recommendations(1))

algo_svd = SVD()
algo_svd.fit(trainset)

predictions_svd = algo_svd.test(testset)
accuracy.rmse(predictions_svd)

def get_svd_recommendations(user_id, top_n=10):
    user_inner_id = algo_svd.trainset.to_inner_uid(user_id)
    user_ratings = algo_svd.trainset.ur[user_inner_id]
    recommendations = []
    for iid in algo_svd.trainset.all_items():
        if iid not in [item[0] for item in user_ratings]:
            est_rating = algo_svd.predict(user_id, algo_svd.trainset.to_raw_iid(iid)).est
            recommendations.append((iid, est_rating))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return [algo_svd.trainset.to_raw_iid(iid) for iid, _ in recommendations[:top_n]]

print(get_svd_recommendations(1))

def recommend_movies(user_id, top_n=10):
    user_recommendations = get_user_recommendations(user_id, top_n)
    item_recommendations = get_item_recommendations(user_id, top_n)
    svd_recommendations = get_svd_recommendations(user_id, top_n)

    recommendations = {
        'User-User': user_recommendations,
        'Item-Item': item_recommendations,
        'SVD': svd_recommendations
    }

    return recommendations

print(recommend_movies(1))