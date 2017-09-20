# This script contains the init and run functions needed to 
# deploy the trained model as a web service.
# Test the functions before deploying the web service.

def init():
    # One-time initialization of PySpark and predictive model
    import pyspark
    from azureml.assets import get_local_path
    from mmlspark.TrainClassifier import TrainedClassifierModel

    global trainedModel
    global spark
    spark = pyspark.sql.SparkSession.builder.appName("Adult Census Income").getOrCreate()

    # load the model file using link file reference
    local_path = get_local_path('mmlspark_model/AdultCensusModel.link')
    print(local_path)
    trainedModel = TrainedClassifierModel.load(local_path)
   
def run(input_df):
    # Compute prediction
    prediction = trainedModel.transform(input_df)
    return prediction.first().scored_labels

if __name__ == "__main__":
    # Test scoring
    init()
    sample = spark.createDataFrame([('10th','Married-civ-spouse',35.0)],[' education',' marital-status',' hours-per-week'])
    print('Positive vs negative prediction: ',run(sample))