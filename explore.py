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
    fig, ax = plt.subplots(1,1, figsize=(20,12))

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
    sns.scatterplot(ax=ax1, y='logerror', x='sqft_lotsize_cluster',hue='sqft_lotsize_cluster',
                    palette='colorblind', data=train)
    ax1.legend().remove()
    sns.scatterplot(ax=ax2, y='sqft', x='lotsizesquarefeet', hue='sqft_lotsize_cluster',
                    palette='colorblind', data=train)
    plt.show()

def location_home_size_cluster_plot(train):
    sns.set(font_scale=1.5)
    sns.set_style('white')

    fig, ax = plt.subplots(2,2, figsize=(20,12))
    fig.tight_layout(pad=4.0)
    sns.scatterplot(ax=ax[0,0], y='sqft', x='latitude', hue='location_home_size_cluster',
                    palette='colorblind', data=train)
    ax[0,0].legend().remove()
    sns.scatterplot(ax=ax[0,1], y='sqft', x='longitude', hue='location_home_size_cluster',
                    palette='colorblind', data=train)
    ax[0,1].legend(loc='upper right')
    sns.scatterplot(ax =ax[1,0], y='logerror', x='location_home_size_cluster',
                    palette='colorblind', data=train, hue='location_home_size_cluster')
    ax[1,0].legend().remove()
    sns.scatterplot(ax=ax[1,1], data =train, x='longitude', palette='colorblind', y= 'latitude', hue='location_home_size_cluster')
    ax[1,1].legend().remove()

    plt.show()

def location_total_size_cluster_plot(train):
    sns.set(font_scale=1.5)
    sns.set_style('white')

    fig, ax = plt.subplots(1,1, figsize=(12,5))
    fig.tight_layout(pad=4.0)
    sns.scatterplot(y='logerror', x='location_total_size_cluster',
                    palette='colorblind', data=train)
    plt.show()

def special_features_cluster_plot(train):
    sns.set(font_scale=1.5)
    sns.set_style('white')

    fig, ax = plt.subplots(1,1, figsize=(12,5))
    sns.scatterplot(y='logerror', x='special_features_cluster',
        palette='colorblind', data=train)
    plt.show()