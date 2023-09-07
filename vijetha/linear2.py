import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\MY PC\Downloads\insurance.csv")
df1 = pd.DataFrame(data)

# Scatter plot of age vs charges
plt.scatter(df1['age'], df1['charges'], color='red', label='Data Points')

# Linear regression
regr = linear_model.LinearRegression()
x = np.asanyarray(df1['age']).reshape(-1, 1)
y = np.asanyarray(df1['charges'])
regr.fit(x, y)

# Plotting the regression line
plt.plot(x, regr.predict(x), color='blue', linewidth=3, label='Regression Line')

# Predicting and plotting user's age
d = int(input('Enter your age\n'))
predicted_charge = regr.predict([[d]])[0]
plt.scatter(d, predicted_charge, color='green', marker='x', s=100, label='Predicted Value')

plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Linear Regression: Age vs Charges')
plt.legend()
plt.show()

print('Your predicted insurance charges according to the age: {:.2f}'.format(predicted_charge))
