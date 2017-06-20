# For more details on configuring execution targets, go to: http://aka.ms/vienna-docs-exec

# Run adult_census.py in a local Docker container.
az ml execute start -t docker train_mmlspark.py 0.1

# Create myvm.compute file to point to a remove VM
az ml computecontext attach --name <myvm> --address <ip address or FQDN> --username <username> --password <pwd>

# Run adult_census.py in a Docker container (with Spark) in a remote VM:
az ml execute start -t myvm train_mmlspark.py 0.3

# Create myhdi.compute to point to an HDI cluster
az ml computecontext attach --name <myhdi> --address <ip address or FQDN of the head node> --username <username> --password <pwd> --cluster

# Run it in a remote HDInsight cluster:
az ml execute start -t myhdi train_mmlspark.py 0.5