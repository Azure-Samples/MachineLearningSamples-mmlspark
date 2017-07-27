# This PySpark code uses regular Spark ML constructs. 
# It is a lot more involved comparing to the MMLSpark version.

import numpy as np
import pandas as pd
import pyspark
import os
import requests

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

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
run_logger.log("Regularization Rate", reg)

# create a new Logistic Regression model.
lr = LogisticRegression(regParam=reg)

# string-index and one-hot encode the education column
si1 = StringIndexer(inputCol=' education', outputCol='ed')
ohe1 = OneHotEncoder(inputCol='ed', outputCol='ed-encoded')

# string-index and one-hot encode the matrial-status column
si2 = StringIndexer(inputCol=' marital-status', outputCol='ms')
ohe2 = OneHotEncoder(inputCol='ms', outputCol='ms-encoded')

# string-index the label column into a column named "label"
si3 = StringIndexer(inputCol=' income', outputCol='label')

# assemble the encoded feature columns in to a column named "features"
assembler = VectorAssembler(inputCols=['ed-encoded', 'ms-encoded', ' hours-per-week'], outputCol="features")

# put together the pipeline
pipe = Pipeline(stages=[si1, ohe1, si2, ohe2, si3, assembler, lr])

# train the model
model = pipe.fit(train)

# make prediction
pred = model.transform(test)

# evaluate. note only 2 metrics are supported out of the box by Spark ML.
bce = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction')
au_roc = bce.setMetricName('areaUnderROC').evaluate(pred)
au_prc = bce.setMetricName('areaUnderPR').evaluate(pred)

print("Area under ROC: {}".format(au_roc))
print("Area Under PR: {}".format(au_prc))

# Log the metrics
run_logger.log("AU ROC", au_roc)
run_logger.log("AU PRC", au_prc)

print("******** SAVE THE MODEL ***********")
model.write().overwrite().save("./outputs/AdultCensus.mml")
