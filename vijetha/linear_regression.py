import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
data=pd.read_csv('insurance.csv')
dfl=pd.DataFrame(data)
print(dfl)
dfl.isnull()