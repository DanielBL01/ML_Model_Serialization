'''

    Use the model.pickle which is now a byte stream which
    is simply an ordered sequence of byte characters which 
    can be deserialized to retrieve our trained machine learning
    model that was created in train.py

'''

import pickle
import pandas as pd

file_path = 'melb_data.csv'
data = pd.read_csv(file_path)

data = data.dropna(axis=0)

features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = data[features]

pickle_in = open("model.pickle", "rb")

model = pickle.load(pickle_in)
print("Prediction: {}".format(model.predict(X.head())))

