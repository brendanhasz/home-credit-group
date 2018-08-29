# Load packages
import numpy as np
import pandas as pd
import featuretools as ft
from featuretools import selection 

# Load applications data
train = pd.read_csv('../application_train.csv')
test = pd.read_csv('../application_test.csv')
bureau = pd.read_csv('../bureau.csv')
bureau_balance = pd.read_csv('../bureau_balance.csv')
cash_balance = pd.read_csv('../POS_CASH_balance.csv')
card_balance = pd.read_csv('../credit_card_balance.csv')
prev_app = pd.read_csv('../previous_application.csv')
payments = pd.read_csv('../installments_payments.csv')

# JUST FOR TESTING
train = train.loc[:10000,:]
test = test.loc[:10000,:]
bureau = bureau.loc[:10000,:]
bureau_balance = bureau_balance.loc[:10000,:]
cash_balance = cash_balance.loc[:10000,:]
card_balance = card_balance.loc[:10000,:]
prev_app = prev_app.loc[:10000,:]
payments = payments.loc[:10000,:]

# Define entities
# Each entry is "Name", (df, "id_col_name")
entities = { #use id_col_name not in df for new index, w/ None uses 1st col
    'app': (app, 'SK_ID_CURR'),
    'bureau': (bureau, 'SK_ID_BUREAU'),
    'bureau_balance': (bureau_balance, 'New'),
    'cash_balance': (cash_balance, 'New'),
    'card_balance': (card_balance, 'New'),
    'prev_app': (prev_app, 'SK_ID_PREV'),
    'payments': (payments, 'New') 
}

# Define relationships between dataframes
# Each entry is (parent_entity, parent_variable, child_entity, child_variable)
relationships = [
    ('app', 'SK_ID_CURR', 'bureau', 'SK_ID_CURR'),
    ('bureau', 'SK_ID_BUREAU', 'bureau_balance', 'SK_ID_BUREAU'),
    ('app', 'SK_ID_CURR', 'prev_app', 'SK_ID_CURR'),
    ('app', 'SK_ID_CURR', 'cash_balance', 'SK_ID_CURR'),
    ('app', 'SK_ID_CURR', 'payments', 'SK_ID_CURR'),
    ('app', 'SK_ID_CURR', 'card_balance', 'SK_ID_CURR')
]

# Define which primitives to use
agg_primitives =  ['count', 'mean', 'num_unique', 'percent_true']
trans_primitives = ['time_since_previous']

# Run deep feature synthesis
t0 = time.time()
dfs_feat, dfs_defs = ft.dfs(entities=entities,
                            relationships=relationships,
                            target_entity='app',
                            trans_primitives=trans_primitives,
                            agg_primitives=agg_primitives, 
                            verbose = True,
                            max_depth=2, n_jobs=2)
print('DFS took %0.3g sec' % (time.time()-t0))

# Save generated to file
dfs_feat.to_csv('feature_matrix.csv', index = False)
