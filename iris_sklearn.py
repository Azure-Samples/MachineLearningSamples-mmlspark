# Please make sure scikit-learn is included the conda_dependencies.yml file.

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle
import sys
from azureml_sdk import data_collector

run_logger = data_collector.current_run() 

print ('Python version: {}'.format(sys.version))
print ()

# load Iris dataset
iris = load_iris()
print ('Iris dataset shape: {}'.format(iris.data.shape))

# load features and labels
X, Y = iris.data, iris.target

# train a logistic regression model
# C is the inverse of the regularization strenght. Change this value to get a different accuracy.
clf1 = LogisticRegression(C=100).fit(X, Y)
print (clf1)

# record accuracy
accuracy = clf1.score(X, Y)
print ("Accuracy is {}".format(accuracy))
run_logger.metrics.custom_scalar("Accuracy", accuracy)

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
X_new = [[3.0, 3.6, 1.3, 0.25]]
print ('New sample: {}'.format(X_new))
pred = clf2.predict(X_new)
print ('Predicted class: {}'.format(pred))
run_logger.metrics.custom_scalar("Predicted Class", pred)