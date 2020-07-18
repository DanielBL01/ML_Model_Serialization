'''

    We deserialize the pickle file that has our 
    saved trained forest model to predict the same 
    dataset about the melbourne data. We also use the 
    mean_absolute_error to get an accurate measurement of
    how our machine learning model predicts the dataset

'''

import pickle 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

file_path = 'melb_data.csv'
data = pd.read_csv(file_path)

data = data.dropna(axis=0)

y = data.Price

features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

pickle_in = open("better_model.pickle", "rb")

model = pickle.load(pickle_in)

pred = model.predict(val_X)
print("MAE: {}".format(mean_absolute_error(val_y, pred)))
