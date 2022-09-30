# I was getting name error so i had to change the import pandas
# to import pandas as pd and so changed line 15 to pd.__version__

# Python version
import sklearn
import pandas as pd
import numpy
import scipy
import sys
print('Python: {}'.format(sys.version))
# scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
print('numpy: {}'.format(numpy.__version__))
# pandas
print('pandas: {}'.format(pd.__version__))
# scikit-learn
print('sklearn: {}'.format(sklearn.__version__))

# print "Hello, World"
print("Hello, World!")
