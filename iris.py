from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle
import sys

print ('Python version: {}'.format(sys.version))
print ()

# load Iris dataset
iris = load_iris()
print ('Iris dataset shape: {}'.format(iris.data.shape))

# load features and labels
X, Y = iris.data, iris.target

# train a logistic regression model
clf1 = LogisticRegression(C=150).fit(X, Y)
print (clf1)

# serialize the model on disk
print ("Export the model to model.pkl")
f = open('model.pkl', 'wb')
pickle.dump(clf1, f)
f.close()

# load the model back in memory
print("Import the model from model.pkl")
f2 = open('model.pkl', 'rb')
clf2 = pickle.load(f2)

# predict a new sample
X_new = [[5.0, 3.6, 1.3, 0.25]]
print ('New sample: {}'.format(X_new))
print ('Predicted class: {}'.format(clf2.predict(X_new)))
