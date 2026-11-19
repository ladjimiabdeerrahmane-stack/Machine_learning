import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

table = pd.read_csv("Salary.csv")
x = table["YearsExperience"].values
y = table["Salary"].values
plt.title("Salary in years")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

# Calculate the y = ax+b for the dataset to find the lowest errors on the plot
# Method 1 : 

X = x.mean()
Y = y.mean()

a = ((x - X) * (y - Y)).sum() / ((x - X) ** 2).sum()
b = Y - a * X

y_line = a * x + b

MSE = ((y - y_line) ** 2).sum() / len(x)
R2 = 1 - ((y - y_line) ** 2).sum() / ((y - Y) ** 2).sum()

print(f"a = {a:.4f}, b = {b:.4f}")
print(f"MSE = {MSE:.4f}")
print(f"R^2 = {R2*100:.4f} %")

plt.scatter(x, y , label = "Datas")
plt.plot(x, y_line, color="green" , label = "Regression line 1")

# Method 2 :

def cost_function(x , y , w , b):
    m = len(x)
    sum_cost = 0
    for i in range(m):
        f = w*x[i]+b
        sum_cost += (f - y[i])**2
    total_cost = sum_cost *(1/(2*m))
    return total_cost

def gradial_function(x , y , w , b):
    dc_dw = 0
    dc_db = 0
    m = len(x)
    for i in range(m):
        f = w*x[i]+b
        dc_dw += (f - y[i])*x[i]
        dc_db += (f - y[i])
    dc_dw = dc_dw * (1/m)
    dc_db = dc_db * (1/m)    
    return dc_dw , dc_db

def gradient_decent( x , y , alpha , iteractions):
    w = 0
    b = 0
    for i in range(iteractions):
        dc_dw , dc_db = gradial_function(x , y , w , b)
        w = w - alpha * dc_dw
        b = b - alpha * dc_db
        print(f"iteraction : {i} , cost : {cost_function(x , y , w , b)}")
    return w , b

alpha = 0.01
iteractions = 10000
final_w , final_b = gradient_decent(x , y , alpha , iteractions)
print(f"w = {final_w:.4f} , b = {final_b:.4f}")

X = np.linspace(min(x) , max(x),100)
Y = final_w * X + final_b
plt.plot(X, Y, color = "red", label = "Regression line 2")
plt.legend()
plt.show()
