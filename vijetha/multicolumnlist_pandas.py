import pandas as pd
data={'student_number':[12,45,67,89,20],
       'marks_obtained':[89,90,45,67,65],
       'scored_class':[1,2,3,4,5]}
table_data=pd.DataFrame(data)  #prints indices with each columns
print(table_data)
data=[1,2,3,4,5,6,6,7,8]
series_data=pd.Series(data)   #prints only single column with series
print(series_data)
data_list=[[1,2,3,4],[1,2,3,4],[1,2,3,4]]  #prints using rows and columns indices
table_data=pd.DataFrame(data_list)
print(table_data[1][2])
print(table_data.iloc[1][2]) #iloc means first see row ,here 1st rows 2nd value
print(table_data.iloc[0])  #if nothing is there then see 1st column 2nd row
print(table_data.head(2))   #print 1st 2 list
print(table_data.tail(2))  #print last two list
data_list=[1,2,3,4],[5,6,7],[9,10,11]
table_data=pd.DataFrame(data_list)
null_values=table_data.isnull()
print(null_values)
print(table_data.dropna())