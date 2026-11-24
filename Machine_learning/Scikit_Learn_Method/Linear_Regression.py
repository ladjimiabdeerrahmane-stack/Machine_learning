import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score

df = pd.read_csv("ML_Data.csv")
x = df['years_experience'].values.reshape(-1 ,1)
y = df['annual_salary'].values

x_train , x_test , y_train , y_test = train_test_split(x , y , random_state = 11 , test_size = 0.2)

mod = LinearRegression()
mod.fit(x_train , y_train)

a = mod.coef_[0]
b = mod.intercept_

y_pred = mod.predict(x_test)

print("MSE =", mean_squared_error(y_test , y_pred))
print("R2  =",r2_score(y_test , y_pred))

xs = np.linspace(x.min() , x.max() , 100)   # sorted grid: plot() draws in array order
plt.scatter(x , y , s = 4 , alpha = 0.2 , color = "blue" , label = "Data")
plt.plot(xs , a*xs+b , color = "red" , label = "Linear curve")
plt.title("Annual Salary based on YOE" , color = "green")
plt.xlabel("Years of Experience" , color = "Blue")
plt.ylabel("Annual Salary" , color = "red")
plt.legend()
plt.show()