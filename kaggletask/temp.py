 
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
import sklearn
import re
#from sklearn_extra.cluster import KMedoids


df_main=pd.read_csv(r'S:\vinodtask/tz2.csv',header=0,thousands=',')

df=df_main.copy()



df['respondtime']=df['desc'].str.extract('(....-..-.. @ ..:..:..)', expand=True)
df['respondtime']=df['respondtime'].str.replace('@','').str.strip()
df[['desc1','desc2','desc3']]=df['desc'].str.split(';',expand=True,n=2)
df['station']=df['desc3'].str.replace('(....-..-.. @ ..:..:..)', '')
df['station']=df['station'].str.replace("-", '').str.replace(";", '').str.strip()

#df.to_csv(r'E:\vinodtask/tz4.csv')

responsetime=pd.to_datetime(df['respondtime'], format='%Y-%m-%d  %H:%M:%S')
inputtime=pd.to_datetime(df['timeStamp'], format='%d-%m-%Y %H:%M',errors='coerce')
timeelapsed= pd.DataFrame((inputtime - responsetime).dt.total_seconds(),columns=['timeleapsed'])



df['twp'].value_counts().head(5)

df['title'].unique()
df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])
df['Reason'].value_counts()

#df.loc[(df['desc3'].str.contains('@')),'desc6']= df.loc[(df['desc3'].str.contains('@')),'desc3']
#df_cluster=sklearn.preprocessing.scale(df_cluster,with_mean=True,with_std=True)


df_cluster= pd.concat([df[['lat','lng','zip']],timeelapsed],axis=1)
df_cluster= df_cluster.dropna()

df= df[(df['lat']>35) & (df['lat']<45)]
df_cluster=df_cluster.reset_index(drop=True)
kmeans = KMeans(n_clusters=5, random_state=0).fit(df_cluster)
labeltitle='withtimeelapsed'

label = pd.DataFrame(kmeans.labels_,columns=[labeltitle])
centres= kmeans.cluster_centers_

cluster= pd.concat([df_cluster,label],axis=1)


axs=plt.figure()
plt.title('distribution of points')
plt.xlabel('latitude')
plt.ylabel('longitude')
for i in range(5):
    plt.plot(cluster[cluster[labeltitle]==i]['lat'],cluster[cluster[labeltitle]==i]['lng'], 'o',markersize=5)
    
   
for i in range(5):   
    plt.plot(centres[i][0],centres[i][1], 'x',markersize=7)
    print(len(cluster[cluster[labeltitle]==i]))

#model=KMedoids(metric="manhattan", n_clusters=5)
#cls=model.fit(np.array(df2[['lat','lng']]))
#labels= model.predict(np.array(df2[['lat','lng']]))
#centroids = model.cluster_centers_