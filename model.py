# !/usr/bin/env python3
#  -*- coding: utf-8 -*-
# title           :model.py
# description     : take in a csv file and a pretrained model and a training predictors name pickle file and output RMSE(root mean square error of prediction)
# author          :Vivek Khetan
# date            :19022018
# version         :0.1
# notes           : needs to have trained model weights and predictor name list
# python_version  :3.6.3
# ==============================================================================


def predict(pickle_model, pickle_colnames,  test_df ):
    """
    Input:
        - pickle_model: saved and trained pickle model
        - pickle_colname: saved predictor names used for training the model
        - test_df: dataframe we want to use for testing
    Output:
        - RMSE value: using provided and predicted hammer price
    """
    
    import pickle
    from sklearn.metrics import mean_squared_error
    import random
    import sys
    import numpy as np
    import pandas as pd 
    import sklearn
    
    
    # test data reading
    df = pd.read_csv(test_df, encoding = "latin-1", 
                 parse_dates=['auction_date', 'artist_birth_year', 'artist_death_year', 'year_of_execution'],
                infer_datetime_format = True)
    ## Test data cleaning
    df.replace(to_replace='Circa\\r\\t\\t\\t\d\d\d\d', value=0, regex=True, inplace = True)
    df.replace(to_replace='n.d.', value=0, regex=False, inplace = True)

    df['auction_afer_death'] = (df['auction_date'].dt.year - df['artist_death_year'].dt.year)
    df['auction_afer_birth'] = (df['auction_date'].dt.year - df['artist_birth_year'].dt.year)
    df['art_age'] = df['auction_date'].dt.year - pd.to_datetime(df['year_of_execution'], errors='coerce').dt.year

    df['auction_afer_death'] = df['auction_afer_death'].fillna((df['auction_afer_death'].mean()))
    df['auction_afer_birth'] = df['auction_afer_birth'].fillna((df['auction_afer_birth'].mean()))
    df['art_age'] = df['art_age'].fillna((df['art_age'].mean()))
    df.drop(['auction_date', 'artist_death_year', 'artist_birth_year','year_of_execution' ], axis=1, inplace=True)
    # ## artist_name is of no use for regression: dropping it
    df.drop(['artist_name', 'edition', 'title'], axis=1, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df.drop(['location', 'materials'], axis=1, inplace=True)
    df = pd.get_dummies(df, prefix = ['currency','nationailty', 'category'], prefix_sep ="_",
               columns = ['currency','artist_nationality', 'category'],
               sparse = False, dummy_na=True, drop_first= True)
    
    
    ## given value of y
    y_test = df[['hammer_price']] # response
    x_test = df.drop(['hammer_price'], axis=1) # predictors

    
    ## name of features provided in the current dataset (along with created dummy variable)
    test_colnames = list(x_test) 
    
    
    ## let's normalise x_test
    

    
    x_test= x_test.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_test_scaled = min_max_scaler.fit_transform(x_test)
    x_test= pd.DataFrame(x_test_scaled, columns=test_colnames)

    
    
    
    # Let's load the predictors lsit used for training the model
    with open(pickle_colnames, 'rb') as file:  
        train_colnames = pickle.load(file)
    # Get missing columns in the training test
    missing_cols = set(train_colnames) - set(test_colnames)
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        x_test[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    x_test = x_test[train_colnames]
    
    
    # Loading the saved trained model file
    with open(pickle_model, 'rb') as file:  
        pickle_model = pickle.load(file)
        
    y_pred = pickle_model.predict(x_test)
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    
    return RMSE
  
#############

RMSE = predict('pickle_model.pkl','train_col.pkl','data.csv' )
RMSE