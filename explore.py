import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

def logerror_dist_by_county(train):
    '''
    Take in zillow train dataframe and plots a histogram
    of the logerror split by county.
    '''
    #set font size
    sns.set(font_scale=2)
    #set plot style
    sns.set_style('white')
    #set size
    fig, ax = plt.subplots(1,1, figsize=(20,8))

    #plot distribution by county
    sns.histplot(train[train.county=='LA'].logerror, kde=True, ax=ax, binwidth=.01, color='#d55e00', label='LA', alpha=0.75)
    sns.histplot(train[train.county=='Orange'].logerror, kde=True, ax=ax, binwidth=.01,color ='#f0e442', label='Orange', alpha=0.75)
    sns.histplot(train[train.county=='Ventura'].logerror, kde=True, ax=ax, binwidth=.01, color='#0072b2', label='Ventura', alpha=0.75)

    #set x-axis limit
    ax.set_xlim(-2,2)
    #set title and show legend
    ax.set_title('Distribution of logerror by County')
    ax.legend()
    plt.show()

def sqft_lotsize_cluster_plot(train):
    '''
    
    '''
    sns.set(font_scale=1.5)
    sns.set_style('white')

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,8))
    fig.tight_layout(pad=4.0)

    ax1.set_title('logerror of Each Centroid in sqft_lotsize_cluster')
    sns.scatterplot(ax=ax1, y='logerror', x='sqft_lotsize_cluster',hue='sqft_lotsize_cluster',
                    palette='colorblind', data=train)
    ax1.legend().remove()
    ax1.set_xlabel('Centroid Number')

    ax2.set_title('House Size vs. Lot Size')
    sns.scatterplot(ax=ax2, y='sqft', x='lotsizesquarefeet', hue='sqft_lotsize_cluster',
                    palette='colorblind', data=train)
    ax2.legend(title='Centroid Number')
    ax2.set_ylabel('House Size (square feet)')
    ax2.set_xlabel('Lot Size (square feet)')

    plt.show()

def location_home_size_cluster_plot(train):
    '''
    
    '''
    sns.set(font_scale=1.5)
    sns.set_style('white')

    fig, ax = plt.subplots(2,2, figsize=(20,16))
    fig.suptitle('Comparing Longitude, Latitude and Square Footage')
    fig.subplots_adjust(hspace=0.3)
    
    ax[0,0].set_title('Square Footage vs. Latitude')
    sns.scatterplot(ax=ax[0,0], y='sqft', x='latitude', hue='location_home_size_cluster',
                    palette='colorblind', data=train)
    ax[0,0].legend().remove()
    ax[0,0].set_ylabel('Square Feet')
    
    ax[0,1].set_title('Square Footage vs. Longitude')
    sns.scatterplot(ax=ax[0,1], y='sqft', x='longitude', hue='location_home_size_cluster',
                    palette='colorblind', data=train)
    ax[0,1].legend(title='Centroid Number', loc='upper right')
    ax[0,1].set_ylabel('Square Feet')

    ax[1,0].set_title('logerror of Each Centroid in location_home_size_cluster')
    sns.scatterplot(ax =ax[1,0], y='logerror', x='location_home_size_cluster',
                    palette='colorblind', data=train, hue='location_home_size_cluster')
    ax[1,0].legend().remove()
    ax[1,0].set_xlabel('Centroid Number')
    

    sns.scatterplot(ax=ax[1,1], data =train, x='longitude', palette='colorblind', y= 'latitude', hue='location_home_size_cluster')
    ax[1,1].legend().remove()
    ax[1,1].set_title('Latitude vs. Longitude (map of area)')

    plt.show()

def location_total_size_cluster_plot(train):
    '''
    
    '''
    sns.set(font_scale=1.5)
    sns.set_style('white')

    fig, ax = plt.subplots(1,1, figsize=(12,5))
    
    ax.set_title('logerror of Each Centroid in location_total_size_cluster')
    sns.scatterplot(y='logerror', x='location_total_size_cluster',
                    palette='colorblind', data=train)
    ax.set_xlabel('Centroid Number')
    plt.show()

def mean_logerror_location_total_size_centroids(train):
    logerror_mean = train.logerror.mean()
    print(f'Mean logerror\ntrain (all): {logerror_mean}\n-------------------')
    for i in range(0,6):
        print(f"Centroid {i} : {train[train['location_total_size_cluster'] ==i].logerror.mean()}\n-------------------")

def location_total_size_cluster_1_ttest(train):
    '''
    
    '''
    logerror_mean = train.logerror.mean()
    location_total_size_cluster_1 = train[train['location_total_size_cluster'] ==1].logerror
    p = stats.ttest_1samp(location_total_size_cluster_1,logerror_mean)
    print(f'p-value: {p}')

def special_features_cluster_plot(train):
    '''
    
    '''
    sns.set(font_scale=1.5)
    sns.set_style('white')

    fig, ax = plt.subplots(1,1, figsize=(12,5))
    
    ax.set_title('logerror of Each Centroid in special_features_cluster')
    sns.scatterplot(y='logerror', x='special_features_cluster',
        palette='colorblind', data=train)
    ax.set_xlabel('Centroid Number')
    plt.show()

def mean_logerror_special_features_centroids(train):
    logerror_mean = train.logerror.mean()
    print(f'Mean logerror\ntrain (all): {logerror_mean}\n-------------------')
    for i in range(0,5):
        print(f"Centroid {i} : {train[train['special_features_cluster'] ==i].logerror.mean()}\n-------------------")


def special_features_cluster_1_ttest(train):
    '''
    
    '''
    logerror_mean = train.logerror.mean()
    special_features_cluster_1 = train[train['special_features_cluster'] ==1].logerror
    p = stats.ttest_1samp(special_features_cluster_1,logerror_mean)
    print(f'p-value: {p}')