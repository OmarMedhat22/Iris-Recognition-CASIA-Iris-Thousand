# Iris-Recognition-CASIA-Iris-Thousand
![](image.jpg)

## Definition

- This project represents a new iris recognition technique that detects and classify the iris images efficiently with high accuracy. The iris recognition model is beginning by eye detection process then the iris detection process takes place which detects the iris inside the eyes then iris segmentation process gets iris images that will be saved and used in the last process which is responsible for iris classification using convolutional neural network.<br/><br/>
- The dataset used in this project is CASIA-Iris-Thousand version 4 which contains 20000 images from 1000  different persons. <br/> <br/>
- The model uses a pre-trained convolutional neural network(CNN) model DenseNet-201 in the last process of iris classification.



![](image2.jpg)
## Description

- ##### eye_detection_2.py: this file represents the first process in this project where it's responsible for eye detection. This stage will ensure that the images contain eyes.

- ##### eyes_iris_detection_2.py: this file represents the second process where it's responsible for iris detection. The model takes the images detected in the last phase and it ensures that there is an iris inside the eyes the image will pass to the next step if it passed this step.

- ##### iris_segmentation_2.py: this file represents the third process where the iris features are extracted.

- ##### iris_extraction_2.py: this file combines all the above files in on file. it begins by eye detection then iris segmentation and saves the output that will be used in the iris classification process.

- ##### iris_classification_2. ipynb: this file represents the last process which is iris classification. this file uses pre-trained DenseNet-201 to classify between 1000 different classes in CASIA-Iris-Thousand version 4 dataset.

-  ##### image_aug.py: this file used for image augmentation but it's not necessary to use it as you can run the project without using this file.

##  How to run this project ?

### Dependencies

#### install this libraries:
- numpy
- keras
- sklearn
- opencv
- glob
- tensorflow

### To run this project you will need to:
1. Download the CASIA-Iris-Thousand dataset from this link [CASIA-Iris-Thousand]( http://www.cbsr.ia.ac.cn/china/Iris%20Databases%20CH.asp)

2. Change your directory names that contain the dataset to the name in the python and notebook files in these  lines:<br/>
    - In iris_extreaction_2.py:
 
         ```html
        #here create directory name that contain the dataset "CASIA-Iris-Thousand/"
        
        for filepath in glob.iglob('CASIA-Iris-Thousand/*'):

        #here create  directory name that contain the the extracted iris feautres "final_ casia/"
        cv2.imwrite('final_ casia/'+str(L)+'.'+str(number)+".jpg",new_roi)

        ```
    - In iris_classification_2.ipynb:

         ```html
        # here directory name is "final_casia" which contain extracted iris features

        
        for filefilepath in glob.iglob('final_ casia/*'):

        ```
3. Run iris_extraction_2.py.

4. Open iris_classification_2.ipynb and run it's cells.


##### I wrote a research paper in this project found [here](https://www.isroset.org/pdf_paper_view.php?paper_id=1775&4-IJSRMS-03115.pdf)
