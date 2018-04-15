# Basic Imports
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## Load Dataset
fname = 'yelp.json'
df    = pd.read_json(fname)
print 'Total number of entries: {}'.format(len(df))


# Convert Select Cols to categories 
catcols = ['business_id','user_id']
for col in catcols:
    df[col] = df[col].astype('category')

df['user_id'] = df.user_id.cat.codes
df['business_id'] = df.business_id.cat.codes

# Convert Date to Better Format
# Using month, day & year
df['month'] = df.date.dt.month
df['day']   = df.date.dt.day
df['year']  = df.date.dt.year
df['dOw']   = df.date.dt.dayofweek
df.drop(columns=['date'])

cols  = ['business_id','user_id','review_id',
         'year','month','day','dOw','cool',
         'funny','useful','text','stars']
df    = df[cols]

print df.head()

## Use Doc2Vec To Vectorize
vmodel = Doc2Vec.load('doc2vec.model')
ids    = df.review_id.values

# Get Vectorized texts
v_data = []
for i in ids:
    v = vmodel.docvecs[i]
    v_data.append(v)

v_data = np.vstack(v_data)

# Append text features to others
fcols  = ['business_id','user_id','year','month','day','dOw','cool','funny','useful']
x_data = df[fcols].values
x_data = np.append(x_data,v_data,axis=-1)
y_data = df['stars'].values

print 'Feature size: ', x_data.shape
print 'Target  size: ', y_data.shape


# Split into Train & Validation Sets
print '\nSplitting into train & valid...'
xtrain,xvalid,ytrain,yvalid = train_test_split(x_data,y_data,
                                             test_size=0.33,random_state=32)

print 'Size of Train set: {}'.format(len(xtrain))
print 'Size of Valid set: {}'.format(len(xvalid))


# Training & Evaluation
print '\nTraining & evaluating model...'
# Fit model no training data
model = XGBClassifier(n_estimators=250,njobs=4,objective='multi:softmax',num_class=5)
model.fit(xtrain, ytrain)
ypred = model.predict(xvalid)
ypred = [round(value) for value in ypred]

# evaluate predictions
acc   = accuracy_score(ytest, ypred)
print("Accuracy: %.2f%%" % (acc * 100.0))



# ## Test on Test Set
# dt    = pd.read_json('yelpHeld.json')
# for col in ['business_id','user_id']:
#     dt[col] = dt[col].astype('category')

# dt['user_id'] = dt.user_id.cat.codes
# dt['business_id'] = dt.business_id.cat.codes

# vt_data = model.docvecs.infer_vector(v) for v in dt. 


# # Generate Submission
# if gen_submission:
#     csvname='submission.csv'
#     ds = pd.DataFrame({'review_id': review_id,
#                        'stars'    : ypred})
#     ds.to_csv(csvname,index=False)
#     print 'Submission file saved as ' + csvname