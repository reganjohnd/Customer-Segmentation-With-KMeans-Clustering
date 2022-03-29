'''
Segment shopping customers

Problem: understand the  target customers for the marketing team to plan a strategy

Context: Your boss wants you to identify the most important shopping groups based on:
    1. Income
    2. Age
    3. Mall shopping score

Objective Market Segmentation: Divide your mall target arket into approachable groups.
Create subsets of a amrket based on demographics beahavioural criteria to better understnd the target for marketing activities

The Approach
1. Perform exploratory data analysis
2. Use KMEANS clustering algorithm to create our segments
3. Use summary statistics on the clusters
4. Visualise
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('customerSegmentation.csv')

### Univariate analysis - one variable

## Get a general look at the data statistics
print(df.describe())

## Study the histograms to see the distribution of the relevant numeric variables
columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in columns:
        plt.figure()
        sns.distplot(df[i])
        plt.show()
    
## See the interaction between the relevant numeric variables and the 2 given genders
for i in columns:
    plt.figure()
    sns.kdeplot(df[i], shade = True, hue = df['Gender'])
    plt.show()

## Study the box plots to understand the relationship between the relevant variables and the 2 given genders
for i in columns:
    plt.figure()
    sns.boxplot(data = df, x = 'Gender', y = df[i])
    plt.show()


## understand the split between male and female in the dataset
gender_counts = df['Gender'].value_counts(normalize = True)
print(gender_counts)


## Bivariate analysis
## Can start to see clusters, can look at different types of scatter plots

## see relationship between Annual Income and Spending score. Clusters start to emerge
sns.scatterplot(data = df, x = 'Annual Income (k$)', y = 'Spending Score (1-100)')
# plt.show()

## See relationship between variables while distinguishing between the 2 given genders
sns.pairplot(df, hue = 'Gender')
plt.show()

## remove customer IDs from dataset
df = df.drop('CustomerID', axis = 1)

## Understand the avergaes for the relevant variables with regards to Age
means_by_age = df.groupby(['Gender'])['Age', 'Annual Income (k$)', 'Spending Score (1-100)'].mean()
print(means_by_age)

## See the correlation matrix for all variables
print(df.corr())
sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')
plt.show()


### Univariate Clustering

## Initiate KMeans object, fit the data with respect to Annual Income, have a look at the clustering labels assigned
clustering1 = KMeans()
clustering1.fit(df[['Annual Income (k$)']])
print(clustering1.labels_)

## Add the Income clustering labels to your dataset
df['Income Cluster'] = clustering1.labels_

## See how many observations each cluster contains
print(df['Income Cluster'].value_counts())

## See the clustering inertia
print(clustering1.inertia_) # represents the distance between centroids

## See the relationship between number of clusters and inertia scores
inertia_scores = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df[['Annual Income (k$)']])
    inertia_scores.append(kmeans.inertia_)
    print(i, ": ", kmeans.inertia_)

## See elbow plot to determine how many clusters to use
plt.plot(range(1, 11), inertia_scores)
plt.show()

## Averages for relevant variables with regards to each Income Cluster
average_by_income_cluster = df.groupby('Income Cluster')['Annual Income (k$)', 'Age', 'Spending Score (1-100)'].mean()
print(average_by_income_cluster)




### bivariate clustering

## Initiate KMeans object, fit the data with respect to Annual Income and Spending Score, have a look at the clustering labels assigned
clustering2 = KMeans(n_clusters = 5)
clustering2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
clustering2.labels_
print(clustering2.labels_)

## Add the Income and Spending clustering labels to your dataset
df['Spending and Income Cluster'] = clustering2.labels_

## See the relationship between number of clusters and inertia scores
inertia_scores2 = []
for i in range(1, 11):
    kmeans2 = KMeans(n_clusters = i)
    kmeans2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    inertia_scores2.append(kmeans2.inertia_)
    print(i, ": ", kmeans2.inertia_)

## See elbow plot to determine how many clusters to use
plt.plot(range(1, 11), inertia_scores2)
plt.show()

## create df with the centroids
centers = pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['x', 'y']

## create plot with clusters including the centroids
plt.figure(figsize = (10, 8))
plt.scatter(data = centers, x = 'x', y = 'y', s = 100, c = 'black', marker = '*')
sns.scatterplot(data = df, x = 'Annual Income (k$)', y = 'Spending Score (1-100)', hue = 'Spending and Income Cluster', palette = 'tab10')
plt.show()
plt.savefig('ClusteringImage.png')

## Gender split by cluster
gender_by_cluster = pd.crosstab(df['Spending and Income Cluster'], df['Gender'], normalize = 'index')
print(gender_by_cluster)

## Average of relevant variables by cluster
average_by_cluster = df.groupby('Spending and Income Cluster')['Age', 'Annual Income (k$)', 'Spending Score (1-100)'].mean()
print(average_by_cluster)


# # multivariate clustering

## Initiate StandardScaler object and transform Gender variable into a useable format
scale = StandardScaler()
dff = pd.get_dummies(df, drop_first = True)

## Select relevant fields for dff
dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Male']]
print(dff.head())

## Transform data to perform analysis
dff = scale.fit_transform(dff)
dff = pd.DataFrame(scale.fit_transform(dff))
print(dff.head())

## See the relationship between number of clusters and inertia scores
inertia_score3 = []
for i in range(1, 11):
    kmeans3 = KMeans(n_clusters = i)
    kmeans3.fit(dff)
    inertia_score3.append(kmeans3.inertia_)
    print(i, ': ', kmeans3.inertia_)

## See elbow plot to determine how many clusters to use
plt.plot(range(1, 11), inertia_score3)
plt.show()

### I still need to complete the Multivariate analysis










