'''

    We use train_test_split so that we have both
    validation data as well as training data. This is 
    to prevent false confidence in our machine learning 
    model.

    Random Forests have better performances over the
    DecisionTreeRegressor algorithm because of its way
    of handling underfitting and overfitting

'''

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

file_path = 'melb_data.csv'
data = pd.read_csv(file_path)

data = data.dropna(axis=0)

y = data.Price

features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)

pickle_out = open("better_model.pickle", "wb")
pickle.dump(forest_model, pickle_out)
pickle_out.close()
