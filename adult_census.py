import numpy as np
import pandas as pd
import pyspark
import os
import requests

import mmlspark
from mmlspark.TrainClassifier import TrainClassifier
from pyspark.ml.classification import LogisticRegression

spark = pyspark.sql.SparkSession.builder.appName("Adult Census Income").getOrCreate()

dataFile = "AdultCensusIncome.csv"
if not os.path.isfile(dataFile):
    r = requests.get("https://amldockerdatasets.azureedge.net/" + dataFile)
    with open(dataFile, 'wb') as f:    
        f.write(r.content)

data = spark.createDataFrame(pd.read_csv(dataFile, dtype={" hours-per-week": np.float64}))
data = data.select([" education", " marital-status", " hours-per-week", " income"])

train, test = data.randomSplit([0.75, 0.25], seed=123)
model = TrainClassifier(model=LogisticRegression(), labelCol=" income", numFeatures=256).fit(train)
print("********* TRAINING DATA ***********")
print(train.limit(10).toPandas())

from mmlspark.ComputeModelStatistics import ComputeModelStatistics
prediction = model.transform(test)
metrics = ComputeModelStatistics().transform(prediction)
print("******** MODEL METRICS ************")
print(metrics.limit(10).toPandas())

print("******** SAVE THE MODEL ***********")
model.write().overwrite().save(".outputs/AdultCensus.mml")
#model.write().overwrite().save("wasb:///models/AdultCensus.mml")


