Launch command line from menu: 
File --> Open Command-Line Interface

Two execution targets are provided by default:
local -- execution happens in the local Python runtime.
docker -- execution happens in a Docker instance (which includes a Spark environment) in the local OS. Note you must have a Docker engine installed, and have you C drive (or wherever your %temp% folder) is) if you are on Windows.

To set a remote linux VM execution target, add a file named myvm.compute in the aml_config folder.
To set a remote HDI execution target, add a file named myhdi.compute in the aml_config folder.

1. iris_sklearn.py uses Python and scikit-learn library
To run iris_sklearn.py in local conda environment:
az ml execute start -t local iris_sklearn.py

To run iris_sklearn.py in a local Docker instance:
az ml execute start -t docker iris_sklearn.py

to run iris_sklearn.py in a docker instance in a remote VM:
az ml execute start -t myvm iris_sklearn.py


2. iris_pyspark.py requires a Spark environment.
To run it in a local Docker instance with Spark:
az ml execute start -t docker iris_pyspark.py

To run it in a Docker instance in a remote VM:
az ml execute start -t myvm iris_pyspark.py

To run it in a remote HDI cluster:
az ml execute start -t myhdi iris_pyspark.py

