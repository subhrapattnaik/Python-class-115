import pandas as pd
import plotly.express as px
import pandas as pd
import csv
import plotly.graph_objects as go

df = pd.read_csv("./data.csv")

score_list = df["Score"].tolist()
accepted_list = df["Accepted"].tolist()

fig = px.scatter(x=score_list, y=accepted_list)
fig.show()



import numpy as np
score_array = np.array(score_list)
accepted_array = np.array(accepted_list)

#Slope and intercept using pre-built function of Numpy
m, c = np.polyfit(score_array, accepted_array, 1)

y = []
for x in score_array:
  y_value = m*x + c
  y.append(y_value)

#plotting the graph
fig = px.scatter(x=score_array, y=accepted_array)
fig.update_layout(shapes=[
    dict(
      type= 'line',
      y0= min(y), y1= max(y),
      x0= min(score_array), x1= max(score_array)
    )
])
fig.show()



import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

#reshape the array using the reshape function from a 3 by 3 matrix to a single array

X = np.reshape(score_list, (len(score_list), 1))
Y = np.reshape(accepted_list, (len(accepted_list), 1))

#use logisticRegression model to fit the data into the model so that it can make 
#predictions with maximum accuracy
lr = LogisticRegression()
lr.fit(X, Y)

#creating a scatter plot
plt.figure()
plt.scatter(X.ravel(), Y, color='black', zorder=20)

#defining the sigmoid function to predict the probablity as output 
def model(x):
  return 1 / (1 + np.exp(-x))

#Using the line space function to evenly space the dots
#linespace returns evenly spaced numbers over a specified interval.
#using ravel function to create a single array(converts 2 arrays into one)
X_test = np.linspace(0, 100, 200)
print(lr.coef_)
print(lr.intercept_)
chances = model(X_test * lr.coef_ + lr.intercept_).ravel()

#plotting the plot with different colors.
#axhline stands for axis horizantal line
plt.plot(X_test, chances, color='red', linewidth=3)
plt.axhline(y=0, color='k', linestyle='-')
plt.axhline(y=1, color='k', linestyle='-')
plt.axhline(y=0.5, color='b', linestyle='--')

# do hit and trial by changing the value of X_test
plt.axvline(x=X_test[180], color='b', linestyle='--')

plt.ylabel('y')
plt.xlabel('X')
plt.xlim(75, 85)
plt.show()



#now , we have got our values of slope and intercept
#we will write a small code where we will give the marks scored by the student
#as input and it will tell us the chances of the student being accepted
#by the college


user_score = float(input("Enter your marks here:- "))
chances = model(user_score * lr.coef_ + lr.intercept_).ravel()[0]
if chances <= 0.01:
  print("The student will not get accepted")
elif chances >= 1:
  print("The student will get accepted!")
elif chances < 0.5:
  print("The student might not get accepted")
else:
  print("The student may get accepted")