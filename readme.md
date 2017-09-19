# Using MMLSpark to classify adult income level

This sample demonstrates the power of simplification by implementing a binary classfier using the popular Adult Census dataset, first with the open-source _mmlspark_ Spark package then comparing that with the standad Spark ML constructs. 

## mmlspark vs. Spark ML
As a quick comparision, here is the one-line training code using _mmlspark_:
```python
model = TrainClassifier(model=LogisticRegression(regParam=reg), labelCol=" income", numFeatures=256).fit(train)
```

And here is the equivalent code in standard Spark ML:
```python
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
```

To learn more about _mmlspark_ Spark package, please visit: http://github.com/azure/mmlspark.

## Run this sample:
Run train_mmlspark.py in a local Docker container.
```
$ az ml experiment submit -c docker train_mmlspark.py 0.1
```

Configure a compute environment `myvm` targeting a Docker container running on a remove VM.
```
$ az ml computetarget attach --name myvm --address <ip address or FQDN> --username <username> --password <pwd> --type remotedocker
```

Run train_mmlspark.py in a Docker container (with Spark) in a remote VM:
```
$ az ml experiment submit -c myvm train_mmlspark.py 0.3
```

Configure a compute environment `myvm` targeting an HDInsight Spark cluster.
```
$ az ml computetarget attach --name myhdi --address <ip address or FQDN of the head node> --username <username> --password <pwd> --type cluster
```

Run it in a remote HDInsight cluster:
```
$ az ml experiment submit -c myhdi train_mmlspark.py 0.5
```
