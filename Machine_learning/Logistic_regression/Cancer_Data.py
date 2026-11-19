import numpy as np                  
import pandas as pd                 
import matplotlib.pyplot as plt     

df = pd.read_csv("Cancer_Data.csv")                  
x = df[["radius_mean","texture_mean"]].values        
y = (df["diagnosis"] == "M").astype(int).values     
m , n = x.shape                                     


def sigmoid(z):
    return 1 / (1 + np.exp(-z))      # squeeze any number into a probability between 0 and 1


def cost_function(X, y, w, b):
    cost_sum = 0                                     

    for i in range(m):                              
        z = np.dot(w, X[i]) + b                     
        g = sigmoid(z)                              

       
        cost_sum += -y[i] * np.log(g) - (1 - y[i]) * np.log(1 - g)

    return (1/m) * cost_sum                       


def gradient_function(X, y, w, b):
    grad_w = np.zeros(n)                       # list of predictions, all 0 at first                         
    grad_b = 0                                  

    for i in range(m):                             
        z = np.dot(w, X[i]) + b                   
        g = sigmoid(z)                          

        grad_b += (g - y[i])                     
        for j in range(n):                        
            grad_w[j] += (g - y[i]) * X[i, j]       

    grad_b = (1/m) * grad_b                         
    grad_w = (1/m) * grad_w

    return grad_b, grad_w                       


def gradient_descent(X, y, alpha, iterations):
    w = np.zeros(n)                              
    b = 0                                         

    for i in range(iterations):                   
        grad_b, grad_w = gradient_function(X, y, w, b)  

        w = w - alpha * grad_w               
        b = b - alpha * grad_b                

        if i % 1000 == 0:                         
            print(f"Iteration {i}: Cost {cost_function(X, y, w, b)}")  
    
    return w, b                                


def predict(X, w, b):
    preds = np.zeros(m)                           

    for i in range(m):                          
        z = np.dot(w, X[i]) + b                  
        g = sigmoid(z)                            

        preds[i] = 1 if g >= 0.5 else 0              # 50% or more -> malignant (1), else benign (0)
    
    return preds


learning_rate = 0.01                                 # alpha: size of each step
iterations = 10000                                  # how many steps to take

final_w, final_b = gradient_descent(x, y, learning_rate, iterations)   # train the model

predictions = predict(x, final_w, final_b)           # predict all patients
accuracy = np.mean(predictions == y) * 100           # % of predictions that match the truth
print(f"training accuracy: {accuracy:.2f}%")


# decision boundary: the line where z = 0, i.e. w0*x + w1*y + b = 0
m = -final_w[0] / final_w[1]                         # slope of that line
c = -final_b / final_w[1]                            # where it crosses the y-axis

xmin, xmax = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5    # plot limits for radius (with a little margin)
ymin, ymax = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5    # plot limits for texture
xd = np.array([xmin, xmax])                          # two x points are enough to draw a line
yd = m * xd + c                                      # y values of the line: y = m*x + c

plt.plot(xd, yd, 'k', ls='--', label='Decision Boundary')       # draw the dashed black line
plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)      # color the area below the line blue
plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)    # color the area above the line orange


plt.scatter(x[y == 0,0], x[y == 0,1], color="blue", label="Benign" , s = 20)      # benign dots
plt.scatter(x[y == 1,0], x[y == 1,1], color="red", label="Malignant" , s = 20)    # malignant dots
plt.legend()                                         # show the labels box
plt.xlabel("radius_mean")                            # x-axis name
plt.ylabel("texture_mean")                           # y-axis name
plt.xlim(xmin,xmax)                                  # fix the visible x range
plt.ylim(ymin,ymax)                                  # fix the visible y range
plt.show()                                           # display the plot