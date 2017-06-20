# run this with just "python run.py". It will fail if you run using az ml execute.

import os

# Parameter sweep on regularization rate
reg = 10
while reg > 0.01:
    os.system('az ml execute start -t docker ./train_mmlspark.py {}'.format(reg))
    reg = reg / 2.0