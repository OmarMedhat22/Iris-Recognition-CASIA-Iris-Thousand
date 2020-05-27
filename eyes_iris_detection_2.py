import cv2
import numpy as np
import glob
import pickle

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


eye_num_2=0
def transform_image(img,threshold):
    
    
    retval, threshold = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)


    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

    open_close = cv2.bitwise_or(opening, closing, mask = None)

    return open_close,opening,closing

imgs = []
label=0
final_output = []
lables = []
'''
for filepath in glob.iglob('test/*'):
    
    
    if filepath[-1] == 'g':

        img	= cv2.imread(filepath)
        img=cv2.resize(img,(200,150))

        img	=	cv2.cvtColor(img,	cv2.COLOR_BGR2GRAY)
        
        imgs.append([img,filepath])
        print(filepath)
        
'''

#'''
for filepath in glob.iglob('CASIA-Iris-Thousand/*'):
    num_in_folder=0

    if label > 500:
    
        for filefilepath in glob.iglob(filepath+'/L/*'):
            if filefilepath[-1] == 'g':

                

        

                img	= cv2.imread(filefilepath)
                img=cv2.resize(img,(400,300))

                img	=	cv2.cvtColor(img,	cv2.COLOR_BGR2GRAY)
                
                imgs.append([img,num_in_folder,label,img])
                print(filefilepath)
                num_in_folder = num_in_folder+1

        #'''
        for filefilepath in glob.iglob(filepath+'/R/*'):
            if filefilepath[-1] == 'g':    
        
                img	= cv2.imread(filefilepath)
                img=cv2.resize(img,(400,300))

                img	=	cv2.cvtColor(img,	cv2.COLOR_BGR2GRAY)
                
                imgs.append([img,num_in_folder,label,img])
                print(filefilepath)
                num_in_folder = num_in_folder+1

    #'''       
    label=label+1
        #print(filefilepath)

iris_num=0
for i,j,L,c in imgs:

    
    circles = cv2.HoughCircles(i, cv2.HOUGH_GRADIENT, 10, 100)

    if circles is not None :
        
        circles = np.round(circles[0, :]).astype("int")
        #print(len(circles))
        #print(y)

        maxiumum_average=10000000000000
        #print(len(circles))
        print(i.shape[0])
        print(i.shape[1])
        print(min(i.shape))


        key=True


     
        for (x, y, r) in circles:

            if x+r<=max(i.shape) and y+r<=max(i.shape)and x-r>0 and y-r>0 and r>20:

                key=False

                new_roi = i[y-r:y+r, x-r:x+r]
                #average = np.average(new_roi)

                if average < maxiumum_average:
                    maxiumum_r = r
                    point_x=x
                    point_y=y
                    #maxiumum_average=average  
                
                #cv2.circle(i, (x, y), r, (0, 0, 0), 4)

        if key:
            print("key opened")

            for (x, y, r) in circles:


                    maxiumu_raduis=-4

                    if r > maxiumu_raduis:
                        maxiumum_r = r
                        point_x=x
                        point_y=y
                        #maxiumum_average=average
                        

                             
                
            
        cv2.circle(i, (point_x, point_y), maxiumum_r, (255, 255, 0), 4)


        print(str(L)+'.'+str(j)+"  =  "+str(average))
        #cv2.circle(i, (point_x_medium, point_y_medium), raduis_medium , (255, 255, 0), 4)


        cv2.imwrite('paper_2/iris/'+str(L)+'.'+str(j)+'.jpg',i)
        iris_num = iris_num+1


            #roi_gray = gray[y:y+h, x:x+w]
            #roi_gray = gray[ey:ey+eh, ex:ex+ew]
            #roi_color = img[ey:ey+eh, ex:ex+ew]


print("total_iris_found = ",iris_num)


print("total images number ",len(imgs))






    
