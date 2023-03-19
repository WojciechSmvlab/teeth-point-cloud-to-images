"""
Convert .ply files into 2D RGB .png images
"""
import numpy as np
import cv2  as cv
import time
import os

PIXELS_TO_MM = 603

def converToRGBVal(x):
    #converts values between 0-1529 to RGB color code
    #maximum value is 1529
    conversionStep = int(x//255)
    
    if conversionStep == 0:
        return [x,0,0]
    elif conversionStep == 1:
        return [255, x-255, 0]        
    elif conversionStep == 2:
        return [255, 255, x-(conversionStep*255)]
    elif conversionStep == 3:
        return [255-(x-(conversionStep*255)), 255, 255]
    elif conversionStep == 4:
        return [0, 255-(x-(conversionStep*255)), 255]        
    elif conversionStep == 5:
        return [0, 0, 255-(x-(conversionStep*255))]
    elif conversionStep < 0 or conversionStep > 5:
        print('converToRGBVal: out of range!',x)
        return [0, 0, 0]
        

def loadPointClud(file_path):
    with open(file_path,'r') as file:
        for line in file.readlines():
            if 'element vertex' in line: break
        file.close()
    row_amount = int(line[15:]) #gets amount of rows from header of .ply file    
    return np.loadtxt(file_path,skiprows=13,max_rows=row_amount)

def createAndSaveBirdViews(XYZcords, RGBvals, filename):
    # filename without extenstion
    ScanWidth = np.max(XYZcords[:,0]) - np.min(XYZcords[:,0]) # x axis
    ScanHeight = np.max(XYZcords[:,1]) - np.min(XYZcords[:,1]) # y axis
    ScanDepth = np.max(XYZcords[:,2])-np.min(XYZcords[:,2]) # z axis

    ImgWidth = int(ScanWidth*PIXELS_TO_MM)
    ImgHeight = int(ScanHeight*PIXELS_TO_MM)

    Img, ImgHMap = np.zeros((ImgHeight,ImgWidth,3), dtype=np.uint8), np.zeros((ImgHeight,ImgWidth,3), dtype=np.uint8)

    print(f"Processing: {filename}")
    TimeStart = time.time()

    for x in range(ImgWidth):  
        for y in range(ImgHeight):
            x_mm_coord = x/PIXELS_TO_MM
            y_mm_coord = (ImgHeight-1-y)/PIXELS_TO_MM #avoiding flipped image
            
            distance = (XYZcords[:,0]-x_mm_coord)**2+(XYZcords[:,1]-y_mm_coord)**2 #computing square of distance
            idx = np.argmin(distance) #searching for closest 3D point
            distance = distance[idx]
          
            if(distance<0.0002): #if distance>threshold - blue/black pixel
                Img[y,x,:] = RGBvals[idx][:]
                ImgHMap[y,x,:] = converToRGBVal((XYZcords[idx,2]/ScanDepth)*1529)
            else:
                Img[y,x,:] = [0,0,255]
                ImgHMap[y,x,:] = [0,0,0]
    
    print(f"{filename} processed in {float(time.time() - TimeStart)} s")

    Img, ImgHMap = cv.cvtColor(Img, cv.COLOR_RGB2BGR), cv.cvtColor(ImgHMap, cv.COLOR_RGB2BGR)
    
    return Img, ImgHMap

########### MAIN ##################################

for filename in os.listdir('./3d_models'):
    if not filename.lower().endswith(".ply"): continue     
    filename = filename.lower().split('.')[0] #deleting extension
        
    PointCloud = loadPointClud(f'./3d_models/{filename}.ply')
        
    SpatialQuery = PointCloud[PointCloud[:,2]>-0.009]
    PointXYZCoords = SpatialQuery[:,:3]
    PointRGBVals = SpatialQuery[:,3:]

    #moving cloud to (0,0,0)   
    PointXYZCoords[:,0] = PointXYZCoords[:,0]-np.min(PointXYZCoords[:,0]) 
    PointXYZCoords[:,1] = PointXYZCoords[:,1]-np.min(PointXYZCoords[:,1])
    PointXYZCoords[:,2] = PointXYZCoords[:,2]-np.min(PointXYZCoords[:,2])
        
    Img, ImgHMap = createAndSaveBirdViews(PointXYZCoords,PointRGBVals,filename)

    cv.imwrite(f'./computed_images/{filename}_bird_view.png', Img)
    cv.imwrite(f'./computed_images/{filename}_hmap.png', ImgHMap)