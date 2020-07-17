'''

    Pickle is very useful for when you are working with machine learning algorithms, 
    where you want to save them to be able to make new predictions at a later time, 
    without having to rewrite everything or train the model all over again.

    For this machine learning directory, we will use the scikit-learn library 
    which has all the neccessary algorithms that we need vs something like 
    Pytorch or TensorFlow

    You normally use a Regressor when you are predicting one number value.
    Here we are doing exactly that: predict the price of each house
'''

import pandas as pd
import pickle
from sklearn.tree import DecisionTreeRegressor

file_path = 'melb_data.csv'
data = pd.read_csv(file_path)

# drop any records which contain NaN (Not a Number) value
data = data.dropna(axis=0)

print(data.columns)
print(data.shape)

# Select the prediction target that the model will predict
y = data.Price

# We don't need to use all the columns. Choose "Features"
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = data[features]

print(X.describe())

# Specify a number for random_state to ensure same results each run
model = DecisionTreeRegressor(random_state=1)

model.fit(X, y)
pickle_out = open("model.pickle", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
