# Mask Coverage Detection


## How is it different? :

The model not only detects the presence of mask but can also alert user if they are not wearing the mask properly (uncovered nose or mouth). This will be of great help in curbing the spread of infection, especially in crowded environments. 



## Intuition: 

COVID-19 has recently become a global pandemic caused by a newly discovered coronavirus. The virus is mainly transmitted through droplets generated when an infected person coughs, sneezes, or exhales. In addition to maintaining a proper social distancing, use of face mask is a key measure in suppressing transmission and saving lives. As per WHO, a mask must cover nose, mouth and chin, however due to the limitation of resources, many people have resorted to the use of non-medical(fabric) masks which often creates breathability and fit issues leading to improper wear of masks - leaving their nose or mouth uncovered.  This increases the chance of infection especially in crowded settings.

Based on the latest statistics as on Jan 2020, there has been a total of 85.2M reported cases with an active death count of 1.84M. Thus in order to ensure proper use of mask in areas such as busy shopping centers, religious buildings, restaurants, schools and public transport, having an alerting system to alert whenever a person is not covering their nose or mouth will be of great help in curbing further community spread. 



## Dataset: 
  
[MaskedFace-Net](https://github.com/cabani/MaskedFace-Net) is a dataset of human faces with a correctly or incorrectly worn mask (137,016 images) based on the dataset Flickr-Faces-HQ (FFHQ).

- 67,193 images with Correctly Masked Face Dataset (CMFD) at 1024×1024: [OneDrive (19GB)](https://esigelec-my.sharepoint.com/:f:/g/personal/cabani_esigelec_fr/Ev3GdnQSyzxPjyzU5ElHqagBlkRCaKnnCI85iX-d1L4OHA?e=G7uaYV)

- 66,900 images with Incorrectly Masked Face Dataset (IMFD) at 1024×1024: [OneDrive (19GB)](https://esigelec-my.sharepoint.com/:f:/g/personal/cabani_esigelec_fr/EirjS8ew7-5LnO8I56Uk63wBKebwSlukFBFBaO8N25wn3g?e=Ho1jHG)

> Adnane Cabani, Karim Hammoudi, Halim Benhabiles, and Mahmoud Melkemi, "MaskedFace-Net - A dataset of correctly/incorrectly masked face images in the context of COVID-19", Smart Health, ISSN 2352-6483, Elsevier, 2020. https://doi.org/10.1016/j.smhl.2020.100144 [Preprint version available at arXiv:2008.08016]



## Data Preparation 

- All the images were downscaled to 224*224 to keep the number of parameters to low without compromising much on details.  
- Keras ImageDataGenerator was utilized to artificially expand the size of training data by a series of random translations, rotations. 
- Validation split was set to 30% for model evaluation



## Mask Coverage Model Training: 

Using Transfer Learning, Mobilenetv2 model available with tf.keras api was utilized for training on the mask dataset. The architecture delivers high accuracy results while keeping the parameters and mathematical operations as low as possible to bring deep neural networks to mobile devices.This further aids in speed optimization when compared with other models such as Faster RCNN. 

Keeping the convolutional layers untouched, top fully connected layers were replaced by a new Dense layer of 128 units(relu activation). Dropout of 0.5 was also added to reduce chances of model overfitting. With softmax activation for the last layer, the model probabilites for covered and uncovered class was then obtained.

    mobilenet = MobileNetV2(weights='imagenet',include_top=False,input_shape=(imgsize,imgsize,3)) 
    for layer in mobilenet.layers:
      layer.trainable = False
    
    headModel = mobilenet.output
    headModel = Flatten()(headModel)
    headModel = Dense(128,activation='relu')(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2,activation='softmax')(headModel)
    model = Model(inputs=mobilenet.input, outputs=headModel)

The model was then compiled using an Adam Optimizer with learning rate of 0.0001 and categorical_crossentropy loss function. Since ADAM finds solutions with much larger weights, a weight decay was also added to ensure better generalization and accuracy scores. 

    model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=lr, decay=lr / epch), metrics=['accuracy'])

The model was trained with number of epochs set to 25 and batch size of 32



## Result


**Validation Accuracy: 98.46%**

![Accuracy Plot](/plots/acc_plot_v2.png)

<img src="/plots/stats.png" alt="Precision/Recall" width="600" height="300" />



## Detection on still images and videos


To detect coverage of masks, first a pretrained face detection model was used to detect faces across an image/frame and then mask coverage model was invoked to detect if people are wearing the mask properly or not. 

Some of the pretrained face detection model explored: 

  1. Haar-cascade Detection
    - Rapid Object Detection using a Boosted Cascade of Simple Features
    
  2. Multi-Task Cascaded Convolutional Neural Network(MTCNN)
    - capable of also recognizing other facial features such as eyes and mouth, called landmark detection
    
  3. res10_300x300_ssd_iter_140000.caffemodel
    - DNN face detector is based on the Single Shot Detector (SSD) framework using a ResNet-10 like base Network
    
 
 ***
 
 
 > #### maskCoverageDetector_image.ipynb
 
 For still images, option of using either Haar-cascade or MTCNN is possible. Both models have equivalent performance however MTCNN performs better on most cases.
  
    def detectMask(frame, facemodel, maskmodel,faceModeltype)
    
        # detectMask(frame, faceDetector_type1, maskDetector, 1)  # Haar Cascade
        # detectMask(frame, faceDetector_type2, maskDetector, 2)  # MTCNN
  
  
  
           
 > #### maskCoverageDetector_video.ipynb
 
  For detection of faces in a live Video feed, DNN face detector from OpenCV was used for better speed. A confidence score of 0.6 was used to ensure reduced false positive rate in number of faces detected in each frame.
  
    def detectMask(frame, faceNet, maskNet,conf=0.5)
    
        # detectMask(frame, faceDetector_type3, maskmodel, 0.6)  # Caffe Model
  
  
  
  
## Sample Output



<img src="/results/1.png" alt="Image Output" width="800" height="550" />

[Video Result](/results/rec.mov)




## Current Limitations



In cases of still images or videos, mask coverage model runs on all the detected faces. Thus in scenarios when the face detection model fails to detect a face, the mask coverage model is not invoked, and hence the model returns no output for undetected faces (like in sample output) 
