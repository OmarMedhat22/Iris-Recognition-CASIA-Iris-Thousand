import cv2
import numpy as np
import glob
import pickle

def transform_image(img,threshold):
    
    
    retval, threshold = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)


    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

    open_close = cv2.bitwise_or(opening, closing, mask = None)

    return open_close

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
    
    
    for filefilepath in glob.iglob(filepath+'/L/*'):
        if filefilepath[-1] == 'g':    
    

            img	= cv2.imread(filefilepath)
            img=cv2.resize(img,(200,150))

            img	=	cv2.cvtColor(img,	cv2.COLOR_BGR2GRAY)
            
            imgs.append([img,filefilepath,label])
            print(filefilepath)

    
    for filefilepath in glob.iglob(filepath+'/R/*'):
        if filefilepath[-1] == 'g':    
    

            img	= cv2.imread(filefilepath)
            img=cv2.resize(img,(200,150))

            img	=	cv2.cvtColor(img,	cv2.COLOR_BGR2GRAY)
            
            imgs.append([img,filefilepath,label])
            
    label=label+1
        print(filefilepath)
#'''
print("total images number ",len(imgs))
            
kernel = np.ones((5,5),np.uint8)
import random

random.shuffle(imgs)

test=[]
for i,j,L,c in imgs:
    
    golden_refrence = sum(sum(transform_image(i,0)))
    #print("golden refrence  = "+str(golden_refrence))

    for k in range(10,1000,10):
        
        working_img = transform_image(i,k)
        suming = sum(sum(working_img))
        diffrence = suming-golden_refrence

        if diffrence>800:
            print("the image threshold = " ,k)
            print("the image name " +str(j))
            print(" " )




            _, contours,_ = cv2.findContours(working_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            for z in contours:

                x,y,w,h = cv2.boundingRect(z)
                if x+w<150 and y+h<200 and x-w//4>0:
                    
                    cv2.rectangle(working_img,(x,y),(x+w,y+h),(0,255,0),-2)
                    
            _, contours_2,_ = cv2.findContours(working_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            maxium_area=0
            maxium_area = 0
            maxium_width=0
            point_x=0
            point_y=0
            maxium_height = 0
            for z in contours_2:
                #print(len(i))
                x,y,w,h = cv2.boundingRect(z)
                new_area=h*w
                if x+w<150 and y+h<200 and new_area>maxium_area and x-w//4>0:
                    maxium_area = new_area
                    maxium_width=w
                    point_x=x
                    point_y=y
                    maxium_height = h
                    
                    
                    #cv2.rectangle(working_img,(x,y),(x+w,y+h),(0,255,0),-2)
                    
            #cv2.rectangle(i,(point_x,point_y),(point_x+maxium_width,point_y+maxium_height),(0,255,0),-2)

            center_x = point_x+maxium_width//2
            center_y = point_y+maxium_height//2
            radius = 40

            if center_y-radius>0 and center_x-radius >0  and center_y+radius < 200 and center_x+radius < 150:
                #cv2.circle(c, (int(center_x), int(center_y)), int(radius),  (0, 255, 255), 2)
                new_roi = c[center_y-radius:center_y+radius, center_x-radius:center_x+radius]
                new_roi=cv2.resize(new_roi,(200,150))
                #new_roi	= cv2.cvtColor(new_roi,cv2.COLOR_GRAY2BGR)

                cv2.imwrite('final_ubiris_color/'+str(L)+'.'+str(j)+'.jpg',new_roi)

            #new_roi=cv2.resize(new_roi,(200,150))
            else:
                center_y=c.shape[0]//2
                center_x=c.shape[1]//2
                new_roi = c[center_y-radius:center_y+radius, center_x-radius:center_x+radius]
                new_roi =cv2.resize(new_roi,(200,150))
                #new_roi = cv2.cvtColor(new_roi,cv2.COLOR_GRAY2BGR)

                cv2.imwrite('final_ubiris_color/'+str(L)+'.'+str(j)+'.jpg',new_roi)

            cv2.imwrite('edging_5/'+str(L)+'_'+str(j)+'.jpg',i)
            test.append(i)
            final_output.append(new_roi)
            lables.append(L)



            #cv2.imwrite('edging_5_test/'+str(j[5:]),i)

            break

print("the lenght of final output = ",len(final_output))
print("the of lables = ",len(lables))

final_output=np.array(final_output)
print(final_output.shape)

test=np.array(test)
print(test.shape)

pickle_out = open("test_ubiris.pickle","wb")
pickle.dump(test, pickle_out)
pickle_out.close()

pickle_out = open("ubiris_features.pickle","wb")
pickle.dump(final_output, pickle_out)
pickle_out.close()

pickle_out = open("ubiris_lables.pickle","wb")
pickle.dump(lables, pickle_out)
pickle_out.close()         

