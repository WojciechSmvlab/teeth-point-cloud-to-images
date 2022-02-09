# -*- coding: utf-8 -*-
"""
Convert .ply file to 2 RGB images:
    bird view
    height map
    
Created on Mon Feb  7 09:32:49 2022

@author: Wojciech Szelag


#interesting for 3d clouds https://github.com/marcomusy/vedo
"""
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
from matplotlib import cm
#from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import sys
import cv2   
import time
import os

PIXELS_FOR_MM =603
#PIXELS_FOR_MM =100 #check
SHOW_PLOTS = True

def converToRGBVal(x):
    #converts values between 0-1529 to RGB color code
    #maximum value is 1529
    conversion_step=int(x/255)
    
    if conversion_step==0:
        r=x
        g=0
        b=0
    elif conversion_step==1:
        r=255
        g=x-255
        b=0        
    elif conversion_step==2:
        r=255
        g=255
        b=x-(conversion_step*255)
    elif conversion_step==3:
        r=255-(x-(conversion_step*255))
        g=255
        b=255
    elif conversion_step==4:
        r=0
        g=255-(x-(conversion_step*255))
        b=255        
    elif conversion_step==5:
        r=0
        g=0
        b=255-(x-(conversion_step*255))
    elif conversion_step<0 or conversion_step>5:
        r=0
        g=0
        b=0
        print('converToRGBVal: out of range!',x)
        
    #print('r:',r,' g:',g,' b:',b)
    color=(r,g,b)
    return color

def loadPointClud(file_path):
    file=open(file_path,'r')
    for line in file.readlines():
        if 'element vertex' in line:
            break
    file.close()
    row_amount=int(line[15:]) #gets amount of rows from header of .ply file
    
    point_cloud= np.loadtxt(file_path,skiprows=13,max_rows=row_amount)
    
    return point_cloud

def createAndSaveBirdViews(xyz,filename,z_minimum):
#filename without extenstion
    scan_width=np.max(xyz[:,0])-np.min(xyz[:,0]) #x axis
    scan_height=np.max(xyz[:,1])-np.min(xyz[:,1]) #y axis

    img_width=int(PIXELS_FOR_MM*scan_width)
    img_height=int(PIXELS_FOR_MM*scan_height)


    scan_depth=np.max(xyz[:,2])-np.min(xyz[:,2]) #z axis

    img = np.zeros((img_height,img_width,3), np.uint8)
    img_hmap = np.zeros((img_height,img_width,3), np.uint8)

    print('Images computing for: ',filename)
    start_time = time.time()

    for x in range(img_width):  
        if (x/img_width)%0.025 < 0.01:
            print(int((x/img_width)*100),'%') #progress info  
        for y in range(img_height):
            x_mm_coord=x/PIXELS_FOR_MM
            y_mm_coord=(img_height-1-y)/PIXELS_FOR_MM #avoiding flipped image
            
            idx=np.argmin((xyz[:,0]-x_mm_coord)**2+(xyz[:,1]-y_mm_coord)**2) #searching for closest 3D point
    
            distance=(xyz[idx,0]-x_mm_coord)**2+(xyz[idx,1]-y_mm_coord)**2 #computing square of distance
          
            if(distance<0.0002): #if distance>threshold - blue/black pixel
                img[y,x,0]=rgb[idx][2]
                img[y,x,1]=rgb[idx][1]
                img[y,x,2]=rgb[idx][0]
                
                h_color=converToRGBVal((xyz[idx,2]/scan_depth)*1529)
                img_hmap[y,x,0]=h_color[2]
                img_hmap[y,x,1]=h_color[1]
                img_hmap[y,x,2]=h_color[0]
            else:
                img[y,x,0]=255
                img[y,x,1]=0
                img[y,x,2]=0      
    
                img_hmap[y,x,0]=0
                img_hmap[y,x,1]=0
                img_hmap[y,x,2]=0
    
    print('Computing ',filename,': ',(time.time() - start_time),' seconds')
    
    cv2.imwrite('./computed_images/'+filename+'_bird_view.png', img)
    cv2.imwrite('./computed_images/'+filename+'_hmap_zmin'+str(z_minimum)+'_zrange'+str(scan_depth)+'.png', img_hmap)




########### MAIN ##################################

dirs = os.listdir('./3d_models')

for filename in dirs:
    if ".ply" in filename:       
        filename=filename[0:-4] #deleting extension
        print(filename)
        
        point_cloud=loadPointClud('./3d_models/'+filename+'.ply')
        
        xyz=point_cloud[:,:3]
        rgb=point_cloud[:,3:]
        
        if SHOW_PLOTS:
            ax = plt.axes(projection='3d')
            ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c = rgb/255, s=0.01)
            plt.title(filename + ': Cloud point before crop - 3D view')
            plt.show()
            
            plt.figure()
            plt.plot(xyz[:,1],xyz[:,2])
            plt.title(filename + ': Cloud point before crop - XY view')
            plt.show()
        
        spatial_query=point_cloud[point_cloud[:,2]>-0.009]
        xyz=spatial_query[:,:3]
        rgb=spatial_query[:,3:]
        
        xyz[:,0]=xyz[:,0]-np.min(xyz[:,0]) #moving cloud to (0,0)
        xyz[:,1]=xyz[:,1]-np.min(xyz[:,1]) #moving cloud to (0,0)
        z_minimum=np.min(xyz[:,2])
        xyz[:,2]=xyz[:,2]-z_minimum   #moving cloud to (0,0)
        
        
        if SHOW_PLOTS:
            plt.figure()
            plt.plot(xyz[:,1],xyz[:,2])
            plt.title(filename + ': Cloud point after crop - XY view')
            plt.show()
        
            z_max=np.max(xyz[:,2]) #for height deviation plot
            z_min=np.min(xyz[:,2])
            z_range=z_max-z_min
            
            cmap = cm.get_cmap('nipy_spectral')
            rgb_dev=list()
            rgb_dev=cmap(((xyz[:,2]-z_min)/z_range))
            
            plt.figure()
            plt.scatter(xyz[:,0], xyz[:,1], c = rgb_dev)
            plt.title(filename + ': Cloud point height dev - bird view')
            plt.axis('scaled')
            plt.show()
            
            plt.figure()
            plt.scatter(xyz[:,0], xyz[:,1], c = rgb/255, s=0.1)
            plt.axis('scaled')
            plt.title(filename + ': Cloud point RGB - bird view')
            plt.show()
        
        
        
        #createAndSaveBirdViews(xyz,filename,z_minimum)
