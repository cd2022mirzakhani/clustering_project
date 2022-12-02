import pandas as pd
import numpy as np
import sklearn.metrics as metric

from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans

def create_clusters(train, train_scaled, validate_scaled, test_scaled):
    '''
    Takes in train and train_scaled and creates clusters for the 
    predefined features. Returns train data with clusters as feature columns
    '''
    kmeans_scale5 = KMeans(n_clusters=5, random_state=27)
    kmeans_scale6 = KMeans(n_clusters=6, random_state=27)
    #make cluster for sqft and lotsizesquarefeet
    kmeans_scale5.fit(train_scaled[['sqft', 'lotsizesquarefeet']])
    train['sqft_lotsize_cluster'] = kmeans_scale5.predict(train_scaled[['sqft', 'lotsizesquarefeet']])
    train_scaled['sqft_lotsize_cluster'] = kmeans_scale5.predict(train_scaled[['sqft', 'lotsizesquarefeet']])
    validate_scaled['sqft_lotsize_cluster'] = kmeans_scale5.predict(validate_scaled[['sqft', 'lotsizesquarefeet']])
    test_scaled['sqft_lotsize_cluster'] = kmeans_scale5.predict(test_scaled[['sqft', 'lotsizesquarefeet']])

    #make cluster for latitude, longitude, and sqft
    kmeans_scale6.fit(train_scaled[['latitude', 'longitude', 'sqft']])
    train['location_home_size_cluster'] = kmeans_scale6.predict(train_scaled[['latitude', 'longitude', 'sqft']])
    train_scaled['location_home_size_cluster'] = kmeans_scale6.predict(train_scaled[['latitude', 'longitude', 'sqft']])
    validate_scaled['location_home_size_cluster'] = kmeans_scale6.predict(validate_scaled[['latitude', 'longitude', 'sqft']])
    test_scaled['location_home_size_cluster'] = kmeans_scale6.predict(test_scaled[['latitude', 'longitude', 'sqft']])
    
    #make cluster for sqft, lotsizesquarefeet, latitude and longitude
    kmeans_scale6.fit(train_scaled[['sqft', 'lotsizesquarefeet', 'latitude', 'longitude']])
    train['location_total_size_cluster'] = kmeans_scale6.predict(train_scaled[['sqft', 'lotsizesquarefeet', 'latitude', 'longitude']])
    train_scaled['location_total_size_cluster'] = kmeans_scale6.predict(train_scaled[['sqft', 'lotsizesquarefeet', 'latitude', 'longitude']])
    validate_scaled['location_total_size_cluster'] = kmeans_scale6.predict(validate_scaled[['sqft', 'lotsizesquarefeet', 'latitude', 'longitude']])
    test_scaled['location_total_size_cluster'] = kmeans_scale6.predict(test_scaled[['sqft', 'lotsizesquarefeet', 'latitude', 'longitude']])
    
    #make cluster for 'pool','deck','garage','hottub', 'fireplace'
    kmeans_scale5.fit(train_scaled[['pool','deck','garage','hottub', 'fireplace']])
    train['special_features_cluster'] = kmeans_scale5.predict(train_scaled[['pool','deck','garage','hottub', 'fireplace']])
    train_scaled['special_features_cluster'] = kmeans_scale5.predict(train_scaled[['pool','deck','garage','hottub', 'fireplace']])
    validate_scaled['special_features_cluster'] = kmeans_scale5.predict(validate_scaled[['pool','deck','garage','hottub', 'fireplace']])
    test_scaled['special_features_cluster'] = kmeans_scale5.predict(test_scaled[['pool','deck','garage','hottub', 'fireplace']])

    return train, train_scaled, validate_scaled, test_scaled

def baseline_models(y_train, y_validate):
    '''
    Takes in y_train and y_validate and returns a df of 
    baseline_mean and baseline_median and how they perform
    '''
    train_predictions = pd.DataFrame(y_train)
    validate_predictions = pd.DataFrame(y_validate)
    
    y_pred_mean = y_train.mean()
    train_predictions['y_pred_mean'] = y_pred_mean
    validate_predictions['y_pred_mean'] = y_pred_mean
    
    y_pred_median = y_train.median()
    train_predictions['y_pred_median'] = y_pred_median
    validate_predictions['y_pred_median'] = y_pred_median

    # create the metric_df as a blank dataframe
    metric_df = pd.DataFrame(data=[
    {
        'model': 'mean_baseline', 
        'RMSE_train': metric.mean_squared_error(
            y_train,
            train_predictions['y_pred_mean']) ** .5,
        'RMSE_validate': metric.mean_squared_error(
            y_validate,
            validate_predictions['y_pred_mean']) ** .5,
        'Difference': (( metric.mean_squared_error(
            y_train,
            train_predictions['y_pred_mean']) ** .5)-(metric.mean_squared_error(
            y_validate,
            validate_predictions['y_pred_mean']) ** .5))
    }])

    return metric_df.append(
            {
                'model': 'median_baseline', 
                'RMSE_train': metric.mean_squared_error(
                    y_train,
                    train_predictions['y_pred_median']) ** .5,
                'RMSE_validate': metric.mean_squared_error(
                    y_validate,
                    validate_predictions['y_pred_median']) ** .5,
                'Difference': (( metric.mean_squared_error(
                    y_train,
                    train_predictions['y_pred_median']) ** .5)-(metric.mean_squared_error(
                    y_validate,
                    validate_predictions['y_pred_median']) ** .5))
            }, ignore_index=True)