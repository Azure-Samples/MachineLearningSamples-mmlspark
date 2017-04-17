# For more details on configuring execution targets, go to: http://aka.ms/vienna-docs-exec

# Run iris_sklearn.py in local conda environment.
az ml execute start -t local iris_sklearn.py

# Run iris_sklearn.py in a local Docker container.
az ml execute start -t docker iris_sklearn.py

# Run iris_sklearn.py in a Docker container in a remote machine.
# Note you need to create/configure myvm.compute.
az ml execute start -t myvm iris_sklearn.py

# Run iris_pyspark.py in a local Docker container (with Spark):
az ml execute start -t docker iris_pyspark.py

# Run iris_pyspark.py in a Docker container (with Spark) in a remote VM:
# Note you need to create/configure myvm.compute
az ml execute start -t myvm iris_pyspark.py

# Run it in a remote HDInsight cluster:
# Note you need to create/configure myhdi.compute
az ml execute start -t myhdi iris_pyspark.py