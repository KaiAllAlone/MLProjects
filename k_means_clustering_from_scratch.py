#Univariate Model with 3 centroids
import numpy as np
import seaborn as sn
import matplotlib.pyplot as mp
import pandas as pd
from sklearn.datasets import make_blobs
def initialize_start(data,centroids):
    #data consists of an array of numbers
    centroids['one'] = data[np.random.choice(data.shape[0])]
    max_dist_2=-1
    max_dist_2_num=0
    max_dist_3_num=0
    max_dist_3=-1
    #applying k-means++ algorithm
    for num in data:
        dist=((num[0]-centroids['one'][0])**2 + (num[1]-centroids['one'][1])**2)
        if dist>max_dist_2:
            max_dist_2=dist
            max_dist_2_num=num
    centroids['two']=max_dist_2_num
    for num in data:
        dist=((num[0]-centroids['two'][0])**2 + (num[1]-centroids['two'][1])**2)*(num[0]-centroids['one'][0])**2 + (num[1]-centroids['one'][1])**2
        if dist>max_dist_3:
            max_dist_3=dist
            max_dist_3_num=num
    centroids['three']=max_dist_3_num
    return centroids

#centroids=['one':[x1,y1],'two':[x2,y2],'three':[x3,y3]]
def assign_centroids_points(data,centroids,reln,cent_names):
    one=np.array([])
    two=np.array([])
    three=np.array([])
    for num in data:
        min_dist=1000000
        min_dist_centroid=''
        for centroid in cent_names:#runs 3 times #centroid='one'
            if (((num[0]-centroids[centroid][0])**2)+((num[1]-centroids[centroid][1])**2))**(1/2)<min_dist:
                min_dist_centroid=centroid
                min_dist=(((num[0]-centroids[centroid][0])**2)+((num[1]-centroids[centroid][1])**2))**(1/2)          
        reln[min_dist_centroid]=np.vstack([reln[min_dist_centroid], num])
    return reln
  #reln format{'one':[[centridx,centroidy]]}            
  
def recalculate_centroids(data,centroids,reln,cent_names):
    for centroid in centroids:
          x_mean=np.mean(reln[centroid][:,0])
          y_mean=np.mean(reln[centroid][:,1])
          centroids[centroid]=[x_mean,y_mean]
    return centroids

def main():
    centroids={'one':[],'two':[],'three':[]}
    cent_names=['one','two','three']
    reln={'one':np.array([[0,0]]),'two':np.array([[0,0]]),'three':np.array([[0,0]])}
    # data=np.reshape(np.random.randn(500),(-1,2))
    data, labels = make_blobs(n_samples=300, centers=3, cluster_std=1)
    centroids=initialize_start(data,centroids)
    c=0
    while True:
    #    #keep training until no significant change in centroid positions
        reln=assign_centroids_points(data,centroids,reln,cent_names)
        old_centroids=centroids
        centroids=recalculate_centroids(data,centroids,reln,cent_names)
        if(old_centroids['one']==centroids['one'] and old_centroids['two']==centroids['two'] and old_centroids['three']==centroids['three']):
          break
    palette={'one':'red','two':'blue','three':'green'}
    dt_x=data[:,0]
    dt_y=data[:,1]
    group=np.array([])
    for (idx,num) in enumerate(data):
      if( [dt_x[idx],dt_y[idx]] in reln['one']):
          group=np.append(group,'one')
      elif( [dt_x[idx],dt_y[idx]] in reln['two']):
          group=np.append(group,'two')
      elif( [dt_x[idx],dt_y[idx]] in reln['three']):
          group=np.append(group,'three')        
    df={'x':dt_x,'y':dt_y,'group':group}
    df=pd.DataFrame(df)
    sn.scatterplot(x=df['x'],y=df['y'],hue=df['group'])
    mp.show()
    
main()
                  
        
      
          
          
      
            
        
    
    
    