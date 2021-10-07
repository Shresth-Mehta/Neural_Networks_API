import numpy as np
import random 
import matplotlib.pyplot as plt
import sklearn 
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn import datasets
import pandas as pd


class KMeans_:
  def __init__(self,n_clusters,in_centeroid,epochs,initial,p,use_weights,beta):
    self.n_clusters=n_clusters
    self.clusters=[]
    self.in_centeroid=in_centeroid
    self.epochs=epochs
    self.initial=initial
    self.p=p
    self.use_weights=use_weights
    self.w=None
    self.beta=beta

  
  def __add_cluster(self,head_point):
    self.clusters.append(Cluster(head_point)) 

  
  def __get_distance(self,dp1,dp2):
    if(self.use_weights==True):
      return np.sum(((self.w)**beta)*np.abs(dp1-dp2)**self.p)
    return np.sum(np.abs(dp1-dp2)**self.p)

  
  def pred(self,datapoint):
    return np.argmin([self.__get_distance(datapoint, cluster.centeroid) for cluster in self.clusters])

  
  def __update_weights(self):
    return

    
  def fit(self,X):
    if(self.initial=='iK'):
      if(self.use_weights==True):
        self.w=np.random.random(X[0].shape)
      invalid_points=[]
      num_itr=1
      while(len(invalid_points)!=len(X)):
        max_distance=-1
        max_index=-1
        for i,datapoint in enumerate(X):
          if(i not in invalid_points):
            dis=self.__get_distance(datapoint,self.in_centeroid)
            if(dis>=max_distance):
              max_distance=dis
              max_index=i
        self.__add_cluster(X[max_index])
        self.clusters[-1].empty()
        prev_centeroid=self.clusters[-1].centeroid
        while(True):
          invalid_2=[]
          for i,datapoint in enumerate(X):
            if(i not in invalid_points):
              dis_cluster=self.__get_distance(datapoint,self.clusters[-1].centeroid)
              dis_in_centeroid=self.__get_distance(datapoint,self.in_centeroid)
              if(dis_in_centeroid>=dis_cluster):
                self.clusters[-1].add_datapoint(datapoint)
                invalid_2.append(i)
          current_centeroid=self.clusters[-1].calc_mean()
          #update the weights
          if(self.use_weights==True):
            self.__update_weights()
          
          if(np.all(prev_centeroid==current_centeroid)):
            for j in invalid_2:
              invalid_points.append(j)
            break
          else:
            prev_centeroid=current_centeroid
            self.clusters[-1].empty()
      if(self.n_clusters!=None and len(self.clusters)>self.n_clusters):
        #select n greatest clusters and remove the rest 
        self.clusters.sort(key=lambda p:p.get_size(),reverse=True)
        rem_index=-1
        for index in range(len(self.clusters)):
          if(index>self.n_clusters-1):
            rem_index=index
            break
        del self.clusters[rem_index:]
      for cluster in self.clusters:
        cluster.calc_mean()
        cluster.empty() 

    invalid_points=[]
    old_means=[]
    for cluster in self.clusters:
      old_means.append(cluster.centeroid)
    
    for epoch in range(self.epochs):
      for i,datapoint in enumerate(X):
        self.clusters[self.pred(datapoint)].add_datapoint(datapoint)
      new_means=[]
      
      for cluster in self.clusters:
        new_means.append(cluster.calc_mean())
        cluster.empty()
      same=0
      
      for mean_1,mean_2 in zip(new_means,old_means):
        if(np.all(mean_1==mean_2)):
          same+=1
      old_means.clear()
      
      for mean in new_means:
        old_means.append(mean)  
      
      if(self.use_weights==True):
        self.__update_weights()
      
      if(same==len(new_means)):
        print("Converged after:",epoch,"epochs")
        break
        
  
  def get_cluster_elements(self,X,cluster_number):
    ans=[]
    for datapoint in X:
      if(cluster_number==self.pred(datapoint)):
        ans.append(datapoint)
    return ans
  
  
  def cluster_centers(self):
    ans=[]
    for cluster in self.clusters:
      ans.append(cluster.centeroid)
    return ans
  
  
  def labels(self,X):
    ans=[]
    for datapoint in X:
      ans.append(self.pred(datapoint))
    return ans
  
  
  def accuracy(self,Y,X):
    pred_diff=list(Y-self.labels(X))
    correct=pred_diff.count(0)
    accuracy=correct*100/len(Y)
    return accuracy


class Cluster:
  def __init__(self,head_point):
    self.centeroid=head_point
    self.cluster_members=[head_point]
  
  
  def calc_mean(self):
    sum=0
    for f_vector in self.cluster_members:
      sum+=f_vector
    if(self.get_size()!=0):
      self.centeroid=np.divide(sum,self.get_size())
    else: print("size=0")
    return self.centeroid
  
  
  def add_datapoint(self,datapoint):
    self.cluster_members.append(datapoint)
  
  
  def empty(self):
    self.cluster_members=[]
  
  
  def get_size(self):
    return len(self.cluster_members)


def data_mean(b):
  grand_mean=0
  for point in b:
    grand_mean+=point
  return grand_mean/len(b)

# Script:
# Getting Data:

iris=datasets.load_iris()
x=scale(iris.data)
y=pd.DataFrame(iris.target)
variable_names=iris.feature_names
a=np.reshape(x,(150,4,1))
b=list(a)
iris_df=pd.DataFrame(iris.data)
iris_df.columns=['Sepal_Length','Sepal_Width','Petal_length','Petal_Width']
y.columns=['target']
c=iris.target

# Library Implementation
clustering=KMeans(n_clusters=3,random_state=5)
clustering.fit(x)

#My Implementation
my_clustering=KMeans_(3,data_mean(b),100,'iK',2,False,0)
my_clustering.fit(b)

for i in range(3):
  print(len(my_clustering.get_cluster_elements(b,i)))


# Plotting:

relabel_my=np.choose(r,[2,1,0]).astype(np.int64)
relabel=np.choose(iris.target,[1,0,2]).astype(np.int64)
color_theme=np.array(['darkgray','lightsalmon','powderblue'])
plt.subplot(1,3,1)
plt.scatter(x=iris_df.Petal_length,y=iris_df.Petal_Width,c=color_theme[relabel],s=50)
plt.title('Ground Truth Classification')
plt.subplot(1,3,2)
r=my_clustering.labels(b)
plt.scatter(x=iris_df.Petal_length,y=iris_df.Petal_Width,c=color_theme[relabel_my],s=50)
plt.subplot(1,3,3)
plt.scatter(x=iris_df.Petal_length,y=iris_df.Petal_Width,c=color_theme[clustering.labels_],s=50)

# Accuracy:
g=list(relabel-relabel_my)
print("accuracy:",g.count(0)/len(g))