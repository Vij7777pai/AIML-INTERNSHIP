import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Realestate.csv')
df1 = pd.DataFrame(data)

# Print the DataFrame
print(df1)
df1.info()
df1.describe()

plt.xlabel('X2 house age')
plt.ylabel('Y house price of unit area')
plt.title('X2 house age vs Y house price of unit area')
plt.scatter(df1['X2 house age'], df1['Y house price of unit area'], color='teal')
df1.isnull()
df1.dropna()
plt.show()

regr=linear_model.LinearRegression()
y=np.asanyarray(df1['X2 house age'])
x=np.asanyarray(df1['Y house price of unit area'])
print(x)
print(y)
X=x.reshape(-1,1)
out=regr.fit(X,y)
plt.plot(x,regr.predict(X),color="g") 


print('Please enter your X2 house age')
d=int(input('Enter the value:'))

b=regr.predict([[d]])[0]
b=round(b,2)
print('Your predicted  Y house price of unit area according to the X2 house age {}'.format(b))
plt.show()