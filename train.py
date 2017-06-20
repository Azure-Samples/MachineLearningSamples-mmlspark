import numpy as np
import pandas as pd
import pyspark
import os
import requests

from pyspark.ml.classification import LogisticRegression
import mmlspark
from mmlspark.TrainClassifier import TrainClassifier
from mmlspark.ComputeModelStatistics import ComputeModelStatistics

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

# Use TrainClassifier in mmlspark to train a logistic regression model. Notice that we don't have to do any one-hot encoding, or vectorization. 
# mmlspark does those tasks for us.
model = TrainClassifier(model=LogisticRegression(), labelCol=" income", numFeatures=256).fit(train)

# predict on the test dataset
prediction = model.transform(test)

# compute model metrics
metrics = ComputeModelStatistics().transform(prediction)

print("******** MODEL METRICS ************")
print(metrics.limit(10).toPandas())

print("******** SAVE THE MODEL ***********")
model.write().overwrite().save(".outputs/AdultCensus.mml")

# save model in wasb if running in HDI.
#model.write().overwrite().save("wasb:///models/AdultCensus.mml")


