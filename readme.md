# Using MMLSpark to Classify Income Level

This sample demonstrates the power of simplification by implementing a binary classfier
using the popular Adult Census dataset, first with mmlspark library then comparing that with
the standad Spark ML constructs. 

To learn more about mmlspark library, please visit: http://github.com/azure/mmlspark
For more details on configuring execution targets, go to: http://aka.ms/vienna-docs-exec

Run train_mmlspark.py in a local Docker container.
```
$ az ml experiment submit -c docker train_mmlspark.py 0.1
```

Create myvm.compute file to point to a remove VM
```
$ az ml computetarget attach --name <myvm> --address <ip address or FQDN> --username <username> --password <pwd>
```

Run train_mmlspark.py in a Docker container (with Spark) in a remote VM:
```
$ az ml experiment submit -c myvm train_mmlspark.py 0.3
```

Create myhdi.compute to point to an HDI cluster
```
$ az ml computetarget attach --name <myhdi> --address <ip address or FQDN of the head node> --username <username> --password <pwd> --cluster
```

Run it in a remote HDInsight cluster:
```
$ az ml experiment submit -c myhdi train_mmlspark.py 0.5
```
