# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 19:12:26 2019

@author: atvin
"""

 
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans,DBSCAN,OPTICS
import matplotlib.pyplot as plt
import math
import sklearn
import re
import seaborn as sns
from datetime import *
#from sklearn_extra.cluster import KMedoids
#from pyclustering.cluster.kmedoids import kmedoids
import geopy.distance
from mpl_toolkits.basemap import Basemap
import time

#==============================================================================

# Data import and calculations

df_main=pd.read_csv(r'S:\kaggletask\tz2.csv', header=0,thousands=',')

df=df_main.copy()
df= df.dropna().reset_index(drop=True)
#df=df_main[df_main['title'].str.contains("EMS")==True]
df.info()
df.head()
df['title'].unique()
df['zip'].value_counts().head(5) #top 5 zip codes available

df['Reason']=df['title'].str.split(':', expand=True)[0]
df['inputtime']=df['desc'].str.extract('(....-..-.. @ ..:..:..)', expand=True)
df['inputtime']=df['inputtime'].str.replace('@','').str.strip()
df[['desc1','desc2','desc3']]=df['desc'].str.split(';', expand=True, n=2)
df['station']=df['desc3'].str.replace('(....-..-.. @ ..:..:..)', '')
df['station']=df['station'].str.replace("-", '').str.replace(";", '').str.strip()
df['inputtime']=pd.to_datetime(df['inputtime'], format='%Y-%m-%d  %H:%M:%S')
df['timeStamp']=pd.to_datetime(df['timeStamp'], format='%d-%m-%Y %H:%M',errors='coerce')
df['timeelapsed'] = pd.DataFrame((df['inputtime'] - df['timeStamp']).dt.total_seconds(),columns=['timeleapsed'])
df['hour']=df['inputtime'].dt.hour
df['month']=df['inputtime'].dt.month
df['dayofweek']=df['inputtime'].dt.day_name()
dayHour = df.groupby(by=['dayofweek','hour']).count()['Reason'].unstack()

#==============================================================================

# Data cleaning
df=df[df['Reason']=='EMS']

df = df.drop_duplicates().reset_index(drop=True)

df= df[(df['lat']>35) & (df['lat']<43)]

df= df.dropna().reset_index(drop=True)
#==============================================================================

# Data visualisations

sns.countplot(x='Reason',data=df,palette='viridis')
sns.lmplot(x='lat',y='lng',data=df,fit_reg=False,   hue='Reason',scatter_kws={"s": 5})
sns.boxplot(data=df[['lat','lng']])
sns.countplot(x='dayofweek',data=df,hue='Reason',palette='viridis')
sns.countplot(x='hour',data=df,hue='Reason',palette='viridis')
sns.countplot(x='month',data=df,hue='Reason',palette='viridis')
sns.heatmap(dayHour,cmap='coolwarm')


fig, axes = plt.subplots(figsize=(9,12))
axes.scatter(df.lng, df.lat, s=0.1, alpha=0.5, c='r')
plt.show()


fig = plt.figure(figsize=(12, 12))
m = Basemap(projection='lcc', resolution='h',lat_0=40, lon_0=-75, width=1E6, height=1.2E6)
#m = Basemap(resolution='h',llcrnrlon=df['lng'].min(),llcrnrlat=df['lat'].min(),urcrnrlon=df['lng'].max(),urcrnrlat=df['lat'].max())
#m.shadedrelief()
m.etopo()
m.drawcoastlines(color='black')
m.drawcountries(color='black')
m.drawstates(color='black')
m.scatter(df['lng'].values, df['lat'].values, latlon=True,c='red',s=10,cmap='Reds', alpha=0.5)
lons = [-77.194527, -73.935242, -74.871826,-71.382439,-72.699997]
lats = [ 41.203323, 40.730610, 39.833851,42.407211,41.599998]
x,y = m(lons, lats)
m.plot(x, y, 'kx', markersize=10)
 
labels = ['Pennsylvania', 'NY city', 'New Jersey','Massachusetts','Connecticut']
font = {'family': 'serif',
        'color':  'orange',
        'weight': 'normal',
        'size': 16,
        }
for label, xpt, ypt in zip(labels, x, y):
    plt.text(xpt, ypt, label,fontdict=font)

#df.loc[(df['desc3'].str.contains('@')),'desc6']= df.loc[(df['desc3'].str.contains('@')),'desc3']
#df_cluster=sklearn.preprocessing.scale(df_cluster,with_mean=True,with_std=True)

#==============================================================================



df_cluster= df[['lat','lng']]
start_time = time.time()
# your script


#df_cluster=df_cluster.iloc[0:10000]
#kmeans = kmedoids(data=df_cluster,initial_index_medoids=[100,200,300,400,500],metric="manhattan") #.fit(df_cluster)
#dbscan = DBSCAN(eps=10, min_samples=100, algorithm='ball_tree', metric='haversine').fit(df_cluster.values)
#dbscanlabel=dbscan.labels_
optics = OPTICS( min_samples=5000, xi=.05).fit(df_cluster.values)#, min_cluster_size=.05
opticslabel=optics.labels_

elapsed_time = time.time() - start_time
elapsed_time=time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print(elapsed_time)


optics=pd.DataFrame(opticslabel)
optics.columns=['opticslabel']
optics.to_csv(r'S:\vinodtask\optics5000-0.05xi.csv')

#kmeans = KMeans(n_clusters=5, random_state=0).fit(df_cluster.values)
#labeltitle='labels'
#
#label = pd.DataFrame(kmeans.labels_,columns=[labeltitle])
#centres= kmeans.cluster_centers_

cluster= pd.concat([df,optics],axis=1)
 

axs=plt.figure()
plt.title('distribution of points')
plt.xlabel('latitude')
plt.ylabel('longitude')

for i in range(-1,optics.max()[0]+1):
#    i=-1
    plt.plot(cluster[cluster['opticslabel']==i]['lat'], cluster[cluster['opticslabel']==i]['lng'], 'o', markersize=5)
#for i in range(5):
#    plt.plot(cluster[cluster[labeltitle]==i]['lat'], cluster[cluster[labeltitle]==i]['lng'], 'o', markersize=5)
     
   
#for i in range(5):   
#    plt.plot(centres[i][0],centres[i][1], 'xk', markersize=7)
#    print(len(cluster[cluster[labeltitle]==i]))
    

#for i in range(5):
#    plt.plot(cluster[cluster[labeltitle]==i]['lat'],cluster[cluster[labeltitle]==i]['lng'], 'o',markersize=5)

#def finddist(coords_1,coords_2):
#    return(geopy.distance.vincenty(coords_1, coords_2).km)
#
#
#distance=[]
#for i in range(5): 
#    X =pd.DataFrame([cluster[cluster[labeltitle]==i]['lat'], cluster[cluster[labeltitle]==i]['lng']]).transpose()
#    coords_2 = (centres[i][0],centres[i][1])
#    for index,row in X.iterrows():
#        finddist(X,coords_2)



#for i in range(5): 
#    axs=plt.figure()
#    plt.title('histogram of points')
#    plt.xlabel('bins')   
#    plt.hist(cluster[cluster[labeltitle]==i]['timeleapsed'],bins=range(0,2000,50))

#model=KMedoids(metric="manhattan", n_clusters=5)
#cls=model.fit(np.array(df2[['lat','lng']]))
#labels= model.predict(np.array(df2[['lat','lng']]))
#centroids = model.cluster_centers_