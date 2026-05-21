import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

data = pd.read_csv("dataset/Mall_Customers.csv")

print(data.head())

print(data.info())

print(data.describe())

print(data.isnull().sum())

plt.hist(data['Age'], bins=10)

plt.title("Age Distribution")

plt.xlabel("Age")

plt.ylabel("Number of Customers")

plt.show()

plt.scatter(
    data['Annual Income (k$)'],
    data['Spending Score (1-100)']
)

plt.title("Income vs Spending Score")

plt.xlabel("Annual Income")

plt.ylabel("Spending Score")

plt.show()

X = data[['Annual Income (k$)',
          'Spending Score (1-100)']]

wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42
    )

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.show()

kmeans = KMeans(
    n_clusters=5,
    random_state=42
)

y_kmeans = kmeans.fit_predict(X)

plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=y_kmeans,
    cmap='rainbow'
)

plt.title("Customer Segments")

plt.xlabel("Annual Income")

plt.ylabel("Spending Score")

plt.show()

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    c='black'
)

plt.show()