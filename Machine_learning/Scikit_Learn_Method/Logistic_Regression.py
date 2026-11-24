import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("ML_Data.csv")
x = df["years_experience"].values.reshape(-1 , 1)
y = df["hired"].values

x_train , x_test , y_train , y_test = train_test_split(x , y , random_state=11 , test_size=0.2)

mod = LogisticRegression()
mod.fit(x_train , y_train)

a = mod.coef_[0][0]
b = mod.intercept_[0]

y_pred = mod.predict(x_test)
print("Accuracy =",accuracy_score(y_test , y_pred))
print(confusion_matrix(y_test , y_pred))
xs = np.linspace(-20, 80 , 400)          # wider than the data: shows both flat tails
plt.scatter(x , y , color = "blue" , s = 4 , alpha=0.25 , label = "Datas")
plt.plot(xs , 1/(1+np.exp(-(a*xs+b))) , color = "red" , label = "Logistic_Curve" )
plt.ylim(-0.05 , 1.05)
plt.title("Hired Employees based on Years of Experience" , color = "green")
plt.ylabel(">=0.5 hired | <05 not hired" , color = "red")
plt.xlabel("Years of Experience" , color = "blue")
plt.legend()
plt.show()