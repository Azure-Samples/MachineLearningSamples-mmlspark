# run this with just "python run.py". It will fail if you run using az ml execute.

import os

# Sequential parameter sweep on regularization rate. No parallelism. 
reg = 10
while reg > 0.01:
    os.system('az ml execute start -c docker ./train_mmlspark.py {}'.format(reg))
    #os.system('az ml execute start -c docker ./train_sparkml.py {}'.format(reg))
    reg = reg / 2.0
