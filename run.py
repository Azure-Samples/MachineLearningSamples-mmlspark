# run this with just "python run.py". It will fail if you run using az ml execute.

import os

reg = 10
while reg > 0.001:
    os.system('az ml execute start -t myvm ./train.py {}'.format(reg))
    reg = reg / 2.0