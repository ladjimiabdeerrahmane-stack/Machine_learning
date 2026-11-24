import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix , balanced_accuracy_score

df = pd.read_csv('ML_Data.csv')

X = df.drop(columns = ['employee_id' , 'income' , 'hired'])
y = df['hired']

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 11)

tree = DecisionTreeClassifier(random_state = 11).fit(X_train , y_train)
y_pred = tree.predict(X_test)

pruned = DecisionTreeClassifier(criterion = 'entropy' , ccp_alpha = 0.04 , random_state = 11).fit(X_train , y_train)
y_pred2 = pruned.predict(X_test)

#Two changes from the first tree:                                                                                                
#criterion='entropy' — same idea, different impurity measure: H = -∑ᵢ pᵢ log₂ pᵢ. In        
#practice it rarely changes much; it's a different way of scoring the same splits.          
#ccp_alpha=0.04 — cost-complexity pruning, the important one. A fully grown tree minimizes  
#training error but memorizes noise. Pruning minimizes: R(T) + α ⋅ |T|

for name , model , pred in [('tree (gini)' , tree , y_pred) ,
                            ('pruned (entropy, ccp=0.04)' , pruned , y_pred2)]:
    print(name , '| depth =' , model.get_depth() , '| leaves =' , model.get_n_leaves())
    print('  Balanced accuracy:' , round(balanced_accuracy_score(y_test , pred) , 4))
    print('  Confusion matrix (rows = true, columns = predicted):')
    print(confusion_matrix(y_test , pred))
    print()

importance = pd.Series(pruned.feature_importances_ , index = X.columns).sort_values(ascending = False)
print('Feature importance (pruned tree, non-zero only):')
print(importance[importance > 0].round(4).to_string())
