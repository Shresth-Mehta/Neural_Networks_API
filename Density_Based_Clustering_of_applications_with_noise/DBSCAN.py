import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

class Point:
  def __init__(self,x):
    self.value=x
    self.cluster=-2        #-2:non_visited, -1:Noise

class Dbscan:
  def __init__(self,ep,minPts,p):
    self.ep=ep
    self.minPts=minPts
    self.p=p
    self.labels=[]
    self.data_points=[]
  

  def fit(self,X):
    n_clusters=-1
    self.data_points=[]
    for x in X:
      self.data_points.append(Point(x))
    for point in self.data_points:
      if(point.cluster==-2):
        N=self.get_neighbors(point)
        if(len(N)+1<self.minPts):
          point.cluster=-1
        else:
          n_clusters+=1
          point.cluster=n_clusters
          stack=[]
          for n_point in N:
            stack.append(n_point)
          while(len(stack)!=0):
            temp=stack.pop()
            if(temp.cluster<0):
              temp.cluster=n_clusters
              N_2=self.get_neighbors(temp)
              if(len(N_2)+1>=self.minPts):
                for n_temp in N_2:
                  stack.append(n_temp)
    self.labels=[]
    for point in self.data_points:
      self.labels.append(point.cluster)

  
  def __get_distance(self,dp1,dp2):
    return (np.sum(np.abs(dp1-dp2)**self.p))**(1/self.p)
  
  
  def get_neighbors(self,current):
    ans=[]
    for point in self.data_points:
      d=self.__get_distance(current.value,point.value)
      if(d!=0 and d<=self.ep):
        ans.append(point)
    return ans 


  def number_of_clusters(self):
    return len(set(self.labels))


# script:
# Generating Data:

centers=[[1,1],[-1,-1],[1,-1]]
x,labels_true=make_blobs(n_samples=750, centers=centers,cluster_std=0.4,random_state=0)
x=StandardScaler().fit_transform(x)
x_df=pd.DataFrame(x)
x_df.columns=['x1','x2']
color_theme=np.array(["red","green","blue"])
plt.scatter(x_df.x1,x_df.x2,c=color_theme[labels_true],s=50)

# Clustering
my_clustering=Dbscan(0.3,10,2)
my_clustering.fit(x)
pred_true=my_clustering.labels
color_theme=np.array(["black","red","green","blue"])
plt.scatter(x_df.x1,x_df.x2,c=color_theme[pred_true],s=50)


# Printing the number of points per cluster 
for i in set(my_clustering.labels):
  print(pred_true.count(i),i)
print(set(pred_true))