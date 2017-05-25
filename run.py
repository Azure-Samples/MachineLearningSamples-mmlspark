# run iris_sklearn.py with descending regularization rates
# run this with Python, NOT az ml execute if you want to execute this in Docker or remotely.

import os

reg = 10
while reg > 0.005:
    os.system('az ml execute start -t local ./iris_sklearn.py {}'.format(reg))
    reg = reg / 2
    
