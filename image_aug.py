import numpy as np
import glob
import cv2


Folder_name = "augmented_2"

def rotate_image(image,name,number):

    img_rotate_90_clockwise = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    img_rotate_90_clockwise= cv2.resize(img_rotate_90_clockwise,(70,70))
    number = number+20000
    cv2.imwrite("augmented_2/" + str(name) +'.'+str(number)+ '.jpg', img_rotate_90_clockwise)

def flip(img,direction,name,number):
    img_flip_lr = cv2.flip(img, direction)
    if direction==0:
        
        number = number+3001
    else: 
        number = number+10
    cv2.imwrite("augmented_2/" + str(name)+'.'+str(number) + '.jpg', img_flip_lr)

def rotate_image_negative(image,name,number):
    
    img_rotate_90_counterclockwise = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_rotate_90_counterclockwise= cv2.resize(img_rotate_90_counterclockwise,(200,150))
    number = number+40
    cv2.imwrite("augmented_2/" + str(name) +'.'+str(number)+ '.jpg', img_rotate_90_counterclockwise)


def add_light(image, gamma,name,number):
    number = number+50000
    
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    image=cv2.LUT(image, table)
    if gamma>=1:
          cv2.imwrite("augmented_2/" + str(name) +'.'+str(number)+ '.jpg', image)
    else:
          cv2.imwrite("augmented_2/" + str(name) +'.'+str(number)+ '.jpg', image)


imgs=[]
for filefilepath in glob.iglob('final_casia/*'):
    
    if filefilepath[-1] == 'g':
        
        img	= cv2.imread(filefilepath)
        imgs_colored=cv2.imread(filefilepath)
        #img=cv2.resize(img,(200,150))
        #imgs_colored.append(img)

        print(filefilepath)
        #print(filefilepath[19:-6])
        #print(filefilepath[-5])
        split = filefilepath.split(".")
        #print(split)
        print(split[0][12:])
        print(split[1])

        label=split[0][12:]
        example_number = split[1]
        imgs.append([imgs_colored,label,example_number])


print(len(imgs))



# construct the actual Python generator
turn_over = True
for i,j,k in imgs:
    k=int(k)
    #rotate_image(i,j,k)
    #rotate_image_negative(i,j,k)
    add_light(i, 2,j,k)

    #flip(i,0,j,k)


    

 
