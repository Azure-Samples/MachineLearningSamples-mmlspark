# This PySpark code uses mmlspark library.
# It is much simpler comparing to the regular Spark ML version.

import numpy as np
import pandas as pd
import pyspark
import os
import requests

from pyspark.ml.classification import LogisticRegression
import mmlspark
from mmlspark.TrainClassifier import TrainClassifier
from mmlspark.ComputeModelStatistics import ComputeModelStatistics

from azureml.sdk import data_collector

# create the outputs folder
os.makedirs('./outputs', exist_ok=True)

# Initialize the logger
run_logger = data_collector.current_run() 

# Start Spark application
spark = pyspark.sql.SparkSession.builder.appName("Adult Census Income").getOrCreate()

# Download AdultCensusIncome.csv from Azure CDN. This file has 32,561 rows.
dataFile = "AdultCensusIncome.csv"
if not os.path.isfile(dataFile):
    r = requests.get("https://amldockerdatasets.azureedge.net/" + dataFile)
    with open(dataFile, 'wb') as f:    
        f.write(r.content)

# Create a Spark dataframe out of the csv file.
data = spark.createDataFrame(pd.read_csv(dataFile, dtype={" hours-per-week": np.float64}))
# Choose a few relevant columns and the label column.
data = data.select([" education", " marital-status", " hours-per-week", " income"])

# Split data into train and test.
train, test = data.randomSplit([0.75, 0.25], seed=123)

print("********* TRAINING DATA ***********")
print(train.limit(10).toPandas())

reg = 0.1
# Load Regularization Rate from argument
if len(sys.argv) > 1:
    reg = float(sys.argv[1])
print("Regularization Rate is {}.".format(reg))

# Use TrainClassifier in mmlspark to train a logistic regression model. Notice that we don't have to do any one-hot encoding, or vectorization. 
# We also don't need to convert the label column from string to binary. mmlspark does those all these tasks for us.
model = TrainClassifier(model=LogisticRegression(regParam=reg), labelCol=" income", numFeatures=256).fit(train)
run_logger.log("Regularization Rate", reg)

# predict on the test dataset
prediction = model.transform(test)

# compute model metrics
metrics = ComputeModelStatistics().transform(prediction)

print("******** MODEL METRICS ************")
print("Accuracy is {}.".format(metrics.collect()[0]['accuracy']))
print("Precision is {}.".format(metrics.collect()[0]['precision']))
print("Recall is {}.".format(metrics.collect()[0]['recall']))
print("AUC is {}.".format(metrics.collect()[0]['AUC']))

# Log the metrics
run_logger.log("Accuracy", metrics.collect()[0]['accuracy'])
run_logger.log("AUC", metrics.collect()[0]['AUC'])

print("******** SAVE THE MODEL ***********")
model.write().overwrite().save("./outputs/AdultCensus.mml")

# save model in wasb if running in HDI.
#model.write().overwrite().save("wasb:///models/AdultCensus.mml")
