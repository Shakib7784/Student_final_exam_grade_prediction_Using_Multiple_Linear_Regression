import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv("Student_marks.csv")
# # print(df.tail)
# print(df.shape)
# print(df.info())
# print(df.isnull().values.any())

new_data_accurate = df.dropna()
new_data = new_data_accurate.copy()

# new_data.info()
# new_data.shape


converted_data = {
    "first_term":{
        "A": 10,
        "B+":9,
        "B":8,
        "B-":7,
        "C+":6,
        "C":5,
        "C-":4,
        "D+":3,
        "D":2,
        "F":1,
        "I":0
    },
    "mid_term":{
        "A": 10,
        "B+":9,
        "B":8,
        "B-":7,
        "C+":6,
        "C":5,
        "C-":4,
        "D+":3,
        "D":2,
        "F":1,
        "I":0
    },
    "final":{
        "A": 10,
        "B+":9,
        "B":8,
        "B-":7,
        "C+":6,
        "C":5,
        "C-":4,
        "D+":3,
        "D":2,
        "F":1,
        "I":0
    },
     "Total_grade":{
        "A": 10,
        "B+":9,
        "B":8,
        "B-":7,
        "C+":6,
        "C":5,
        "C-":4,
        "D+":3,
        "D":2,
        "F":1,
        "I":0
    }
    

}

new_data.replace(converted_data, inplace=True)
# new_data.head()
new_data['Total_present'] = new_data['Total_no_of_Cheld']-new_data['Total_absences']
# new_data.head()
# new_data.info()

new_data["Total_no_of_Cheld"] = new_data["Total_no_of_Cheld"].astype(int)
new_data["Total_absences"] = new_data["Total_absences"].astype(int)
new_data["Total_present"] = new_data["Total_present"].astype(int)
# print('\n')
# print(new_data)
# print('\n')
# new_data.head()

#training data
X = new_data.drop(["Program","Course_Code",'final','Total_grade','Total_absences'],axis='columns')
print(X.columns)
y=new_data.final
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y, test_size=0.2)

#fitting data
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
accuracy = model.score(X_test, y_test)
print(f"accuracy rate is: {round(accuracy*100,2)}%")


#showing result
result = model.predict([[42,10,9,40]])
final_result = round(result[0])
grade=''
if(final_result==10):
  grade='A'
elif (final_result==9):
  grade="B+"
elif (final_result==8):
  grade='B'
elif (final_result==7):
  grade="B-"
elif (final_result==6):
  grade='C+'
elif (final_result==5):
  grade="C"
elif (final_result==4):
  grade='C-'
elif (final_result==3):
  grade="D"
elif (final_result==2):
  grade='D+'
elif(final_result==1):
  grade="F"
elif(final_result==0):
  grade="I"
print("Your final exam grade is: ",grade)




a={"Total class held":42, "Total Present":40,"first_term":10,"mid_term":9,"final":final_result}
key = list(a.keys())
value = list(a.values())
colors = ['red', 'blue', 'green', 'orange', 'purple']
# Create the bar chart
plt.bar(key, value, color=colors)
# Add a title and axis labels
plt.title('Bar Chart of Student Result Data')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()


predictions = model.predict(X_test)
plt.scatter(y_test,predictions,color="green")
plt.plot(range(0, 10), range(0, 10), '--', color="blue",linewidth=4)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot of Predictions vs. Actual Values')
plt.show()




# print(predictions)







#testing with multiple algorithm
#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
accuracy = model.score(X_test, y_test)
print(f"accuracy rate with Decision Tree  is: {round(accuracy*100,2)}%")

#Random forest 
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,y_train)
accuracy = model.score(X_test, y_test)
print(f"accuracy rate with Random Forest is: {round(accuracy*100,2)}%")

#K Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train,y_train)
accuracy = model.score(X_test, y_test)
print(f"accuracy rate with K Neighbors: {round(accuracy*100,2)}%")

#GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
model.fit(X_train,y_train)
accuracy = model.score(X_test, y_test)
print(f"accuracy rate with Gradient Boosting : {round(accuracy*100,2)}%")