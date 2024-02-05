import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import accuracy_score
import pickle


df = pd.read_csv("transactions.csv")

df.drop(df[df.Category == 'Investments'].index, inplace=True)
df.drop(df[df.Category == 'Income'].index, inplace=True)
df.drop(df[df.Category == 'Amount'].index, inplace=True)


df['Description_value'] = pd.factorize(df['Original Description'])[0]+100
df['Category_value'] = pd.factorize(df['Category'])[0]+900
d = {'debit': '0', 'credit': '1'}
df['Transaction Type'] = df['Transaction Type'].map(d)

features = ['Description_value', 'Transaction Type']
X = df[features]
y = df['Category_value']



# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)



y_pred = dtree.predict(X_test)

acc_randomforest = round(accuracy_score(y_pred,y_test)*100,2)
print("Accuracy: {}".format(acc_randomforest))

""" predict_input_Description = input("Enter Description: ")
predict_input_Description_value = df.loc[df['Original Description'] == predict_input_Description, 'Description_value']

predict_input_Description = input("Enter Description: ")
predict_input_Description_value = df.loc[df['Original Description'] == predict_input_Description, 'Description_value'] """

print(dtree.predict([[101,0]]))

pickle.dump(dtree,open('expensesmodel.sav','wb'))