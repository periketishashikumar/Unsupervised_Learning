import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 
df=pd.read_csv('Mall_Customers.csv') 
x=df.iloc[:,[3,4]].values 
inertias=[]
for i in range(2,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    inertias.append(kmeans.inertia_)
plt.plot(range(2,11), inertias)
plt.title('ELBOW METHOOD')
plt.show()
kmeans=KMeans(n_clusters=5, init='k-means++',random_state=42)
pred=kmeans.fit_predict(x)
print(pred)
plt.scatter(x[pred==0,0], x[pred==0,1],s=100, c='red')
plt.scatter(x[pred==1,0], x[pred==1,1],s=100, c='pink')
plt.scatter(x[pred==2,0], x[pred==2,1],s=100, c='green')
plt.scatter(x[pred==3,0], x[pred==3,1],s=100, c='cyan')
plt.scatter(x[pred==4,0], x[pred==4,1],s=100, c='orange')
plt.show()