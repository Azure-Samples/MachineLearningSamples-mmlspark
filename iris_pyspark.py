import numpy as np
import pandas as pd
import pyspark
import os
import urllib

from pyspark.sql.functions import *
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.ml.feature import *

# start Spark session
spark = pyspark.sql.SparkSession.builder.appName('Iris').getOrCreate()

# download Iris dataset
urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 'iris.csv')
# load iris.csv into Spark dataframe
data = spark.createDataFrame(pd.read_csv('iris.csv', header=None, names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']))
print("First 20 rows of Iris dataset:")
data.show(10)

# vectorize all numerical columns into a single feature column
feature_cols = data.columns[:-1]
assembler = pyspark.ml.feature.VectorAssembler(inputCols=feature_cols, outputCol='features')
data = assembler.transform(data)

# convert text labels into indices
data = data.select(['features', 'class'])
label_indexer = pyspark.ml.feature.StringIndexer(inputCol='class', outputCol='label').fit(data)
data = label_indexer.transform(data)

# only select the features and label column
data = data.select(['features', 'label'])
print("Reading for machine learning")
data.show(10)

# use Logistic Regression to train on the training set
train, test = data.randomSplit([0.75, 0.25])
lr = pyspark.ml.classification.LogisticRegression()
model = lr.fit(train)

# predict on the test set
prediction = model.transform(test)
print("Prediction")
prediction.show(10)

# evaluate the accuracy of the model using the test set
evaluator = pyspark.ml.evaluation.MulticlassClassificationEvaluator(metricName='accuracy')
accuracy = evaluator.evaluate(prediction)

print()
print('#####################################')
print ("Accuracy is {}".format(accuracy))
print('#####################################')
print()
