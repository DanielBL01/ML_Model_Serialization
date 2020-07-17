# Using Pickle and Machine learning

## Why serialize machine learning models?

When you run a python script to train a machine learning model, the completed 
train model only exists until the script terminates. This means that you
would have to write both your code to train and predict in one script. This is not 
idea, especially if you are making a web application that integrates machine 
learning. 

A great way is to "save" a trained machine learning model after you run a 
training script once is with pickle. Pickle allows you to serialize any python
object in a byte stream which is simply just an ordered sequence of byte 
characters that a computer can read and understand. A python object in this 
case can be anything from a dictionary, an object from a class and even
a machine learning model! Once you have serialized your trained model, 
you can deserialize it any time and use your model from a different script!
