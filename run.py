# run iris_sklearn.py with descending regularization rates
# run this with just "python run.py". It will fail if you run using az ml execute.

import os

reg = 10
while reg > 0.005:
    os.system('az ml execute start -t local ./iris_sklearn.py {}'.format(reg))
    reg = reg / 2
    
