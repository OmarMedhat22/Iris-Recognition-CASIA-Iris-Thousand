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
eye_detected = []
iris_eye_detected=[]


#'''
#'''
for filepath in glob.iglob('CASIA-Iris-Thousand/*'):
    num_in_folder=0

    
    
    for filefilepath in glob.iglob(filepath+'/L/*'):
        if filefilepath[-1] == 'g':

            

    

            img	= cv2.imread(filefilepath)
            img=cv2.resize(img,(200,150))

            img	=	cv2.cvtColor(img,	cv2.COLOR_BGR2GRAY)
            
            imgs.append([img,num_in_folder,label,img])
            print(filefilepath)
            num_in_folder = num_in_folder+1

    #'''
    for filefilepath in glob.iglob(filepath+'/R/*'):
        if filefilepath[-1] == 'g':    
    
            img	= cv2.imread(filefilepath)
            img=cv2.resize(img,(200,150))

            img	=	cv2.cvtColor(img,	cv2.COLOR_BGR2GRAY)
            
            imgs.append([img,num_in_folder,label,img])
            print(filefilepath)
            num_in_folder = num_in_folder+1
     #'''       
    label=label+1
        #print(filefilepath)
#'''

print("total images number ",len(imgs))

#'''
eyes_num=0
for i,j,L,c in imgs:

    # cv2.imshow('dd',i)
    i=cv2.resize(i,(400,300))

    eyes = eye_cascade.detectMultiScale(i, 1.01, 0)
    

    if len(eyes)>1:
        eye_detected.append(imgs[eyes_num])
        print(eyes_num)
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







iris_num=0
for i,j,L,c in eye_detected:

    
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
                average = np.average(new_roi)

                if average < maxiumum_average:
                    maxiumum_r = r
                    point_x=x
                    point_y=y
                    maxiumum_average=average  
                
                #cv2.circle(i, (x, y), r, (0, 0, 0), 4)

        if key:
            #print("key opened")

            for (x, y, r) in circles:


                    maxiumu_raduis=-4

                    if r > maxiumu_raduis:
                        maxiumum_r = r
                        point_x=x
                        point_y=y
                        #maxiumum_average=average
                        

                             
                
            
        cv2.circle(i, (point_x, point_y), maxiumum_r, (255, 255, 0), 4)


        #print(str(L)+'.'+str(j)+"  =  "+str(average))
        #cv2.circle(i, (point_x_medium, point_y_medium), raduis_medium , (255, 255, 0), 4)


        cv2.imwrite('paper_3/iris/'+str(L)+'.'+str(j)+'.jpg',i)
        iris_eye_detected.append(eye_detected[iris_num])
        print(iris_num)
        iris_num = iris_num+1


            #roi_gray = gray[y:y+h, x:x+w]
            #roi_gray = gray[ey:ey+eh, ex:ex+ew]
            #roi_color = img[ey:ey+eh, ex:ex+ew]


print("total_iris_found = ",iris_num)


print("total images number ",len(imgs))




imgs= iris_eye_detected

#'''

            
kernel = np.ones((5,5),np.uint8)
import random

random.shuffle(imgs)

test=[]
for i,j,L,c in imgs:
    
    gold,siver,diamond = transform_image(i,0)
    golden_refrence = sum(sum(gold))
    #print("golden refrence  = "+str(golden_refrence))
    found = True

    for k in range(10,10000,10):
        
        working_img,opening,closing = transform_image(i,k)
        suming = sum(sum(working_img))
        diffrence = suming-golden_refrence


        if diffrence>800:
            found = False

            print("the image threshold = " ,k)
            print("the image name " +str(j))
            print(" " )



            cv2.imwrite('paper_2/threshold/'+str(L)+'.'+str(j)+'.jpg',working_img)
            cv2.imwrite('paper_2/opening/'+str(L)+'.'+str(j)+'.jpg',opening)
            cv2.imwrite('paper_2/closing/'+str(L)+'.'+str(j)+'.jpg',closing)
            



            _, contours,_ = cv2.findContours(working_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            for z in contours:

                x,y,w,h = cv2.boundingRect(z)
                if x+w<150 and y+h<200 and x-w//4>0:
                    
                    cv2.rectangle(working_img,(x,y),(x+w,y+h),(0,255,0),-2)
                    cv2.imwrite('paper_2/contour/'+str(L)+'.'+str(j)+'.jpg',working_img)

                    
            _, contours_2,_ = cv2.findContours(working_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            #cv2.imwrite('paper/contour/'+str(L)+'.'+str(j)+'.jpg',contours_2)

            
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

                #cv2.imwrite('paper/threshold/'+str(L)+'.'+str(j)+'.jpg',new_roi)
                cv2.imwrite('final_casia/'+str(L)+'.'+str(j)+'.jpg',new_roi)

            #new_roi=cv2.resize(new_roi,(200,150))
            else:
                center_y=c.shape[0]//2
                center_x=c.shape[1]//2
                new_roi = c[center_y-radius:center_y+radius, center_x-radius:center_x+radius]
                new_roi =cv2.resize(new_roi,(200,150))
                #new_roi = cv2.cvtColor(new_roi,cv2.COLOR_GRAY2BGR)

                cv2.imwrite('final_casia/'+str(L)+'.'+str(j)+'.jpg',new_roi)

            cv2.imwrite('edging_5/'+str(L)+'_'+str(j)+'.jpg',i)
            test.append(i)
            final_output.append(new_roi)
            lables.append(L)

    if  found :
        i =cv2.resize(i,(200,150))
        #i = cv2.cvtColor(i,cv2.COLOR_GRAY2BGR)

    
        #cv2.imwrite('final_iris2/'+str(j[29:]),i)
        cv2.imwrite('final_casia/'+str(L)+'.'+str(j)+'.jpg',new_roi)





            #cv2.imwrite('edging_5_test/'+str(j[5:]),i)

            #break

print("the lenght of final output = ",len(final_output))
print("the of lables = ",len(lables))


