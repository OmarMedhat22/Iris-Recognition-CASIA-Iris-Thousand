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

    if label>=500:
        
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
#'''
eyes_num=0
for i,j,L,c in imgs:

    # cv2.imshow('dd',i)

    
    eyes = eye_cascade.detectMultiScale(i, 1.01, 0)
    

    if len(eyes)>1:
        eyes_num = eyes_num+1

        maxium_area = -3

        for (ex,ey,ew,eh) in eyes:
            area = ew*eh

            if area>maxium_area:
                maxium_area = area
                maxium_width=ew
                point_x=ex
                point_y=ey
                maxium_height = eh
                
            


        cv2.rectangle(i,(point_x,point_y),(point_x+maxium_width,+maxium_height),(255,0,0),2)
        cv2.imwrite('paper_2/eyes/'+str(L)+'.'+str(j)+'.jpg',i)

                #cv2.imwrite('paper/threshold/'+str(L)+'.'+str(j)+'.jpg',working_img)

            #roi_gray = gray[y:y+h, x:x+w]
            #roi_gray = gray[ey:ey+eh, ex:ex+ew]
            #roi_color = img[ey:ey+eh, ex:ex+ew]


print("total_eyes_found = ",eyes_num)
print("total_eyes_found 2 = ",eye_num_2)


print("total images number ",len(imgs))
