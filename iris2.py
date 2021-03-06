from __future__ import print_function
from sklearn.datasets import load_iris
import subprocess
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
#Using panda to read the csv file
df=pd.read_csv('C:/Users/sulav/Downloads/iris-species/Iris.csv')
#printing the first and last five elements on the file using head() and tail() method
#print("* df.head()", df.head(), sep="\n", end="\n\n")
#print("* df.tail()", df.tail(), sep="\n", end="\n\n")
#Printing the unique species 
print("* iris types:", df["Species"].unique(), sep="\n")
#Copying the data into variable df_2
df_2 = df.copy()
#Storing all the unique species name in the variable targets
targets = df_2["Species"].unique()
#Changing the unique species name to int value so that it is easy to process.
map_to_int = {species: n for n, species in enumerate(targets)}
#Adding a column "Target" which has the converted int value
df_2["Target"] = df_2["Species"].replace(map_to_int)

print("* df2.head()", df_2[["Target", "Species"]].head(),
      sep="\n", end="\n\n")
print("* df2.tail()", df_2[["Target", "Species"]].tail(),
      sep="\n", end="\n\n")
print("* targets", targets, sep="\n", end="\n\n")
#Printing the 4 features of the Species
features = list(df_2.columns[1:5])
print("* features:", features, sep="\n")
#Using the value in the column 'Target' and the features in decision tree
y = df_2["Target"]
X = df_2[features]
dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X, y)
print("Accuracy of classifier: ", dt.score(X,y))
#Test sample to make prediction
Z = [[1,3,4,1],[5,4,2,2],[3,4,5,1],[5,4,4,5]]
print ('Mapping of names', df_2["Species"].unique(), 'into integere: ',df_2["Target"].unique())
print ("Test sample: ", Z)
print ("Predicted classification of test sample: ", dt.predict(Z))




