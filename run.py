# run this with just "python run.py". It will fail if you run using az ml execute.

import os

hashSize = 2
while hashSize < 1000:
    os.system('az ml execute start -t myvm ./train.py {}'.format(hashSize))
    hashSize = hashSize * 2