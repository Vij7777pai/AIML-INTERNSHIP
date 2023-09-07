import pandas as pd 
data={'student_number':[12,45,67,89,20],
       'marks_obtained':[89,90,45,67,65],
       'scored_class':[1,2,3,4,5]}
table_data=pd.DataFrame(data)
print(table_data)
data=[1,2,3,4,5,6,6,7,8]
series_data=pd.Series(data)
print(series_data)