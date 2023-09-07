import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV data into a DataFrame
csv_file_path = r'insurance.csv'
data = pd.read_csv(csv_file_path)
df = pd.DataFrame(data)

# Let's say you want to visualize the relationship between age (independent variable) and charges (dependent variable)
x = df['age']  # Independent variable
y = df['charges']  # Dependent variable

# Create a scatter plot
plt.scatter(x, y)
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Scatter Plot of Age vs Charges')
plt.show()

x1 = df['bmi']  # Independent variable
y1 = df['children']  # Dependent variable
plt.scatter(x1, y1)
plt.xlabel('BMI')
plt.ylabel('Children')
plt.title('Scatter Plot of bmi vs children')
plt.show()