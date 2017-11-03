# Using MMLSpark to classify adult income level

This sample demonstrates the power of simplification by implementing a binary classifier using the popular Adult Census dataset, first with the open-source _mmlspark_ Spark package then comparing that with the standard _Spark ML_ constructs. 

## mmlspark vs. Spark ML
As a quick comparision, here is the one-line training code using _mmlspark_, clean and simple:
```python
model = TrainClassifier(model=LogisticRegression(regParam=reg), labelCol=" income", numFeatures=256).fit(train)
```

And here is the equivalent code in standard _Spark ML_. Notice the one-hot encoding, string-indexing and vectorization that you have to do on the training data:
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

## Auto-logging in run history
Metrics can be automatically logged from MMLSpark in Run History with the modules logging package.
When executing `ComputeModelStatistics` function, the metrics will appear in the run automatically:

![ModulesLogging](https://i.imgur.com/UyYOMke.png)

To add the modules logging package:

1. Add a `log4j.properties` file to use the `AmlAppender` and `AmlLayout`

2. Add the modules logging package to `spark_dependencies.yml` file:
```yaml
  - group: "com.microsoft.moduleslogging"
    artifact: "modules-logging_2.11"
    version: "1.0.0024"
```

3.) Configure `log4j` to use the `log4j.properties` file in train_mmlspark.py:
```python
    spark._jvm.org.apache.log4j.PropertyConfigurator.configure(os.getcwd() + "/log4j.properties")
```

## Run this sample:
Run train_mmlspark.py in a local Docker container.
```
$ az ml experiment submit -c docker train_mmlspark.py 0.1
```

Configure a compute environment `myvm` targeting a Docker container running on a remote VM.
```
$ az ml computetarget attach --name myvm --address <ip address or FQDN> --username <username> --password <pwd> --type remotedocker

# prepare the environment
$ az ml experiment prepare -c myvm
```

Run train_mmlspark.py in a Docker container (with Spark) in a remote VM:
```
$ az ml experiment submit -c myvm train_mmlspark.py 0.3
```

Configure a compute environment `myvm` targeting an HDInsight Spark cluster.
```
$ az ml computetarget attach --name myhdi --address <ip address or FQDN of the head node> --username <username> --password <pwd> --type cluster

# prepare the environment
$ az ml experiment prepare -c myhdi
```

Run it in a remote HDInsight cluster:
```
$ az ml experiment submit -c myhdi train_mmlspark.py 0.5
```

## Create a web service using the MMLSpark model
Get the run id of the train_mmlspark.py job from run history.
```
$ az ml history list -o table
```

### Get the model file(s)
And promote the trained model using the run id.

```azurecli
$ az ml history promote -ap ./outputs/AdultCensus.mml -n AdultCensusModel -r <run id>
```
Download the model to a directory.

```azurecli
$ az ml asset download -l ./assets/AdultCensusModel.link -d mmlspark_model
```
**Note**: The download step may fail if file paths within project folder become too long. If that happens, create the project closer to file system root, for example C:/AzureML/Income.

### Create web service schema

Promote the schema file

```azurecli
$ az ml history promote -ap ./outputs/service_schema.json -n service_schema.json -r <run id>
```

Download the schema

```azurecli
$ az ml asset download -l ./assets/service_schema.json.link -d mmlspark_schema
```

### Test the scoring file's init and run functions 

Run score_mmlspark.py in local Docker. Check the output of the job for results.
```
$ az ml experiment submit -c docker score_mmlspark.py
```

### Set the environment
If you have not set up a Model Management deployment environment, see the [Set up Model Managment](https://docs.microsoft.com/azure/machine-learning/preview/deployment-setup-configuration) document under Deploy Models on the documentation page.

If you have already setup an environment, look up it's name and resource group:

```azurecli
$ az ml env list
```

Set the deployment environment:

```azurecli
$ az ml env set -n <environment cluster name> -g <resource group>
```

### Deploy the web service

Deploy the web service

```azurecli
$ az ml service create realtime -f score_mmlspark.py -m mmlspark_model -s mmlspark_schema/service_schema.json -r spark-py -n mmlsparkservice -c aml_config/conda_dependencies.yml
```

Use the Sample CLI command from the output of the previous call to test the web service.

```azurecli
$ az ml service run realtime -i mmlsparkservice -d "{\"input_df\": [{\" hours-per-week\": 35.0, \" education\": \"10th\", \" marital-status\": \"Married-civ-spouse\"}]}"
```
