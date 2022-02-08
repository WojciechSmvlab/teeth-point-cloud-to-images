# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 09:32:49 2022

@author: mvlab
"""

#libraries used
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


#actual code to load,slice and display the point cloud
file_data_path="test_al_2.ply"
point_cloud= np.loadtxt(file_data_path,skiprows=13,max_rows=632782-13-1)

mean_Z=np.mean(point_cloud,axis=0)[2]
spatial_query=point_cloud[abs( point_cloud[:,2]-mean_Z)<1]
xyz=spatial_query[:,:3]
rgb=spatial_query[:,3:]

xyz=point_cloud[:,:3]
rgb=point_cloud[:,3:]

ax = plt.axes(projection='3d')
ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c = rgb/255, s=0.01)
plt.show()

plt.figure()
plt.scatter(xyz[:,0], xyz[:,1], c = rgb/255)
plt.show()

plt.figure()
plt.plot(xyz[:,1],xyz[:,2])
plt.show()

spatial_query=point_cloud[point_cloud[:,2]>-0.01]
xyz=spatial_query[:,:3]
rgb=spatial_query[:,3:]

plt.figure()
plt.plot(xyz[:,1],xyz[:,2])
plt.show()



cmap = cm.get_cmap('inferno')
cmap = cm.get_cmap('nipy_spectral')

z_height_dev=xyz[:,2]

z_max=np.max(z_height_dev)
z_min=np.min(z_height_dev)
z_range=z_max-z_min

rgb_dev=list()
rgb_dev=cmap(((z_height_dev-z_min)/z_range))

plt.figure()
plt.scatter(xyz[:,0], xyz[:,1], c = rgb_dev)
plt.show()

plt.figure()
plt.scatter(xyz[:,0], xyz[:,1], c = rgb/255, s=0.1)
plt.show()

#ciekawa biblio https://github.com/marcomusy/vedo
