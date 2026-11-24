import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix , balanced_accuracy_score

df = pd.read_csv('ML_Data.csv')

X = df.drop(columns = ['employee_id' , 'income' , 'hired'])
y = df['hired']

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 11 , stratify=y)

mod = MinMaxScaler()
X_train = mod.fit_transform(X_train)
X_test  = mod.transform(X_test)
#Why: KNN's decision is based on distance between rows. Without scaling, annual_salary (≈     
#50,000–230,000) would swamp remote_work (0 or 1) and education_level (0–4) — a 1-unit change 
#in salary is meaningless, a 1-unit change in remote_work is everything. MinMax maps every    
#column onto [0, 1]

knn = KNeighborsClassifier(n_neighbors = 9)
knn.fit(X_train , y_train)

y_pred = knn.predict(X_test)

print('Balanced accuracy:', round(balanced_accuracy_score(y_test , y_pred) , 4))
print('Confusion matrix (rows = true, columns = predicted):')
print(confusion_matrix(y_test , y_pred))
