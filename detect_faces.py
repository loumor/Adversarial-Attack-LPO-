# USAGE
# python3 detect_faces.py --image dimTest.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import numpy as np
import argparse
import cv2
import time
import threading      
from multiprocessing.pool import ThreadPool as Pool
from PIL import Image
                                                          

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-iF", "--imageFront", required=True,
    help="path to input image")
ap.add_argument("-iL", "--imageLeft", required=True,
    help="path to input image")
ap.add_argument("-iR", "--imageRight", required=True,
    help="path to input image")
ap.add_argument("-pI", "--pixIncr",type=int, required=True,
    help="The amount of pixel to move the lightspots on the LPO. The smaller the more iterations.")
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())



# Construct the light spot model 
def lightSpotModel(image_dimensionsFLR, face_locationFLR, imagesFLR):

    print("Rescaling Lightspot to for image... [lightSpotModel]")

    basePhotoSize = [1080, 720, 1080, 720] # The original image size the light spot was modeled from 

    #baseFaceSize = [285, 175, 640, 690] # The original face size in the image that the light spot was modeled from

    baseFaceSize = [385, 228, 622, 550] # The origianl face size in the image that the laser spot was modeled from

    # baseLightSpotSize = [100, 88, 100, 88] # The crop image size the light spot lightspot.png

    lightspotFLR = [] # The list of lightspots for each perspective 

    # Loop thru each photo
    for imagePerspective in range(0, len(imagesFLR)):

        face_location = list(face_locationFLR[imagePerspective]) # Convert to list

        # Load spot
        lightspot = cv2.imread("./images/laserspot.png", -1)

        # Check the photo sizes are the same and scale accordingly 
        if (basePhotoSize != image_dimensionsFLR[imagePerspective]):
            #print("Photo to Lightspot Resizing Section:")

            # Find how much the input image varies from the base size 
            imageSizeChange = [(basePhotoSize[0] - image_dimensionsFLR[imagePerspective][0]), (basePhotoSize[1] - image_dimensionsFLR[imagePerspective][1])]
            
            #print("Image Size Difference")
            #print (imageSizeChange)

            # Find how much % the input image varies from the base size 
            percentageDiff = [(imageSizeChange[0]/basePhotoSize[0]), (imageSizeChange[1]/basePhotoSize[1])]

            #print("Percentage diff")
            #print(percentageDiff)

            # Apply scaling 
            width = int(lightspot.shape[1] * (1 - percentageDiff[0]))
            height = int(lightspot.shape[0] * (1 - percentageDiff[1]))
            dim = (width, height)

            #print ("New: Width Height")
            #print (dim)

            # Resize lightspot
            lightspot = cv2.resize(lightspot, dim, interpolation = cv2.INTER_AREA)
    

        
        # Check the face sizes are the same and scale accordingly 
        if (baseFaceSize != face_location):
            #print("Face to Lightspot Resizing Section:")

            # Convert to W H
            baseSizing = [(baseFaceSize[2] - baseFaceSize[0]), (baseFaceSize[3] - baseFaceSize[1])]
            faceSizing = [(face_location[2] - face_location[0]), (face_location[3] - face_location[1])]

            #print("Base Sizing")
            #print(baseSizing)
            #print("Face Sizing")
            #print(faceSizing)

            # Find how much the input image varies from the base size 
            faceSizeChange = [(baseSizing[0] - faceSizing[0]), (baseSizing[1] - faceSizing[1])]

            #print("Image Size Difference")
            #print (faceSizeChange)

            # Find how much % the input image varies from the base size 
            percentageDiff = [(faceSizeChange[0]/baseSizing[0]), (faceSizeChange[1]/baseSizing[1])]

            #print("Percentage diff")
            #print(percentageDiff)

            # Apply scaling 
            width = int(lightspot.shape[1] * (1 - percentageDiff[0]))
            height = int(lightspot.shape[0] * (1 - percentageDiff[1]))
            dim = (width, height)

            #print ("New: Width Height")
            #print (dim)

            # Resize image
            lightspot = cv2.resize(lightspot, dim, interpolation = cv2.INTER_AREA)
            lightspotFLR.append(lightspot)

    return lightspotFLR

    



# Crop the image to just the face and resize 
def cropImage(face_locationFLR, imagesFLR):
    print("Cropping Image... [cropImage]")

    crop_imgFLR = []

    # Loop thru each photo
    for imagePerspective in range(0, len(imagesFLR)):
        image = cv2.imread(imagesFLR[imagePerspective])
        #Crop image around the face positioning. face_location is (startX:startY, endX:endY)    
        crop_imgFLR.append(image[face_locationFLR[imagePerspective][1]:face_locationFLR[imagePerspective][3], face_locationFLR[imagePerspective][0]:face_locationFLR[imagePerspective][2]])

    return crop_imgFLR

    

# Find the lower confidnce positioning for the lightspots 
def LPO_3SPOT_Front(crop_img, lightspot, face_location, image_dimensions, pixelIncrement, net):
    print("Running LPO Front... [LPO_Front]")

    (h_spot,w_spot) = lightspot.shape[:2]

    # Set the range of the offset 
    x_range = face_location[2] - face_location[0] - w_spot
    y_range = face_location[3] - face_location[1] - h_spot


    # The below variable is a shared resource when the program is run in multithread 
    # Becuase its shared the time taken for context switching makes the program run
    # longer in parallel than sequential 
    # The frontOptimize fucntion is also IO bound as we open up an image every iteration 

    # Confidence Score and Image and Number of Iterations 
    imageList = []
    number_of_iterations = []

    def frontOptimize(start, end, multithreaded):
        numberOfInts = 0 # Initialise variable 
        # First lightspot X
        for x_offset_1 in range(start, end, pixelIncrement):
            # First lightspot Y
            for y_offset_1 in range(0, y_range, pixelIncrement):
                # Second lightspot X
                for x_offset_2 in range(0, x_range, pixelIncrement):
                    # Second lightspot Y
                    for y_offset_2 in range(0, y_range, pixelIncrement):
                        # Third lightspot X
                        for x_offset_3 in range(0, x_range, pixelIncrement):
                            # Third lightspot Y
                            for y_offset_3 in range(0, y_range, pixelIncrement):

                                # Attach position offset first lightspot 
                                y1_1, y2_1 = y_offset_1 + face_location[1], y_offset_1 + lightspot.shape[0] + face_location[1]
                                x1_1, x2_1 = x_offset_1 + face_location[0], x_offset_1 + lightspot.shape[1] + face_location[0]

                                #print("X1_1 : Y1_1 : X2_1 : Y2_1")
                                #print(x1_1, y1_1, x2_1, y2_1)

                                # Attach position offset second lightspot 
                                y1_2, y2_2 = y_offset_2 + face_location[1], y_offset_2 + lightspot.shape[0] + face_location[1]
                                x1_2, x2_2 = x_offset_2 + face_location[0], x_offset_2 + lightspot.shape[1] + face_location[0]

                                #print("X1_2 : Y1_2 : X2_2 : Y2_2")
                                #print(x1_2, y1_2, x2_2, y2_2)

                                # Attach position offset third lightspot 
                                y1_3, y2_3 = y_offset_3 + face_location[1], y_offset_3 + lightspot.shape[0] + face_location[1]
                                x1_3, x2_3 = x_offset_3 + face_location[0], x_offset_3 + lightspot.shape[1] + face_location[0]

                                #print("X1_3 : Y1_3 : X2_3 : Y2_3")
                                #print(x1_3, y1_3, x2_3, y2_3)

                                # Adjust alpha levels to extract transparency
                                alpha_s = lightspot[:, :, 3] / 255.0
                                alpha_l = 1.0 - alpha_s
                                
                                image = cv2.imread(args["imageFront"]) 

                                # Apply light spot to cropped image of face 
                                for c in range(0, 3):
                                    image[y1_1:y2_1, x1_1:x2_1, c] = (alpha_s * lightspot[:, :, c] + alpha_l * image[y1_1:y2_1, x1_1:x2_1, c])
                                    image[y1_2:y2_2, x1_2:x2_2, c] = (alpha_s * lightspot[:, :, c] + alpha_l * image[y1_2:y2_2, x1_2:x2_2, c])
                                    image[y1_3:y2_3, x1_3:x2_3, c] = (alpha_s * lightspot[:, :, c] + alpha_l * image[y1_3:y2_3, x1_3:x2_3, c])

                                if (multithreaded == False):
                                    # This line is not thread safe 
                                    numberOfInts = numberOfInts + 1 # Store number of iterations 
                                image, confidence = faceDetection(image, net) # Detect face 

                                # The following lines are for thread safety
                                # Need to append to a local stored list on the thread 
                                # This means the confidence and image is together 
                                # Then store in the global list so that when the threads
                                # join we can iterate thru them and they will be correctly 
                                # associated 

                                # Confidence score and image 
                                confidence_image = []
                                confidence_image.append(confidence)
                                confidence_image.append(image)

                                imageList.append(confidence_image)
        number_of_iterations.append(numberOfInts)

    # Multithreading section 
    '''
    threads = []
    num_splits = 2
    split_size = x_range

    for i in range(num_splits):                                                 
        # determine the indices of the list this thread will handle             
        start = i * split_size                                                  
        # special case on the last chunk to account for uneven splits           
        end = x_range if i+1 == num_splits else (i+1) * split_size                 
        # create the thread                                                     
        threads.append(                                                         
            threading.Thread(target=frontOptimize, args=(start, end, True)))         
        threads[-1].start() # start the thread we just created                  

    # wait for all threads to finish                                            
    for t in threads:                                                           
        t.join()   
    '''
    '''
    pool_size = 1  # your "parallelness

    pool = Pool(pool_size)
    
    split_size = x_range

    for i in range(pool_size):                                                 
        # determine the indices of the list this thread will handle             
        start = i * split_size                                                  
        # special case on the last chunk to account for uneven splits           
        end = x_range if i+1 == pool_size else (i+1) * split_size                 
        # create the thread                                                     
        pool.apply_async(frontOptimize, (start, end, True))                  


    pool.close()
    pool.join() 
    '''
    # Call this to run it in sequential 
    frontOptimize(x_range, y_range, False)

    # Pull the scores out of the list 
    confidenceNumber = []
    for i in range(0, len(imageList)):
        confidenceNumber.append(imageList[i][0])

    # Find the smallest value index 
    lowestConfidenceIndex = confidenceNumber.index(min(confidenceNumber))

    stats = []

    stats.append(number_of_iterations[0])
    stats.append(lowestConfidenceIndex)
    stats.append(confidenceNumber[lowestConfidenceIndex])
    
    #print("FACE LOCATION")
    #print(face_location)
    #print("IMAGE DIM")
    #print(image_dimensions)
    #print("Number of Interations:")
    #print(number_of_iterations[0])    
    #print("Lowest Index:")
    #print(lowestConfidenceIndex)
    #print("Lowest Score:")
    #print(confidenceNumber[lowestConfidenceIndex])

    return imageList[lowestConfidenceIndex][1], stats


# Find the lower confidnce positioning for the lightspots on Left and Right 
def LPO_Front(crop_img, lightspot, face_location, image_dimensions, pixelIncrement, net):
    print("Running LPO Front... [LPO_Front]")

    numberOfInts = 0 # Initialise variable 

    # load our serialized model from disk
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    (h_spot,w_spot) = lightspot.shape[:2]

    # Set the range of the offset 
    x_range = face_location[2] - face_location[0] - w_spot
    y_range = face_location[3] - face_location[1] - h_spot

    # Image lists 
    imageList = []
    

    # First lightspot X
    for x_offset_1 in range(0, x_range, pixelIncrement):
        # First lightspot Y
        for y_offset_1 in range(0, y_range, pixelIncrement):
            
            # Attach position offset first lightspot 
            y1_1, y2_1 = y_offset_1 + face_location[1], y_offset_1 + lightspot.shape[0] + face_location[1]
            x1_1, x2_1 = x_offset_1 + face_location[0], x_offset_1 + lightspot.shape[1] + face_location[0]

            #print("X1_1 : Y1_1 : X2_1 : Y2_1")
            #print(x1_1, y1_1, x2_1, y2_1)

            # Adjust alpha levels to extract transparency
            alpha_s = lightspot[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            
            image = cv2.imread(args["imageFront"]) 

            # Apply light spot to cropped image of face 
            for c in range(0, 3):
                image[y1_1:y2_1, x1_1:x2_1, c] = (alpha_s * lightspot[:, :, c] + alpha_l * image[y1_1:y2_1, x1_1:x2_1, c])

            image, confidence = faceDetection(image, net) # Detect face 

            numberOfInts = numberOfInts + 1
            # Confidence score and image 
            confidence_image = []
            confidence_image.append(confidence)
            confidence_image.append(image)

            imageList.append(confidence_image) 

    # Pull the scores out of the list 
    confidenceNumber = []
    for i in range(0, len(imageList)):
        confidenceNumber.append(imageList[i][0])

    # Find the smallest value index 
    lowestConfidenceIndex = confidenceNumber.index(min(confidenceNumber))
    
    stats = []

    stats.append(numberOfInts)
    stats.append(lowestConfidenceIndex)
    stats.append(confidenceNumber[lowestConfidenceIndex])
    
    #print("FACE LOCATION")
    #print(face_location)
    #print("IMAGE DIM")
    #print(image_dimensions)
    #print("Number of Interations:")
    #print(numberOfInts)
    #print("Lowest Index:")
    #print(lowestConfidenceIndex)
    #print("Lowest Score:")
    #print(confidenceNumber[lowestConfidenceIndex])

    return imageList[lowestConfidenceIndex][1], stats




# Find the lower confidnce positioning for the lightspots on Left and Right 
def LPO_LR(crop_img, lightspot, face_location, image_dimensions, imagesFLR, pixelIncrement, net):
    print("Running LPO Left/Right... [LPO_LR]")

    numberOfInts = 0 # Initialise variable 

    # load our serialized model from disk
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    (h_spot,w_spot) = lightspot.shape[:2]

    # Set the range of the offset 
    x_range = face_location[2] - face_location[0] - w_spot
    y_range = face_location[3] - face_location[1] - h_spot

    # Image lists 
    imageList = []
    

    # First lightspot X
    for x_offset_1 in range(0, x_range, pixelIncrement):
        # First lightspot Y
        for y_offset_1 in range(0, y_range, pixelIncrement):
            
            # Attach position offset first lightspot 
            y1_1, y2_1 = y_offset_1 + face_location[1], y_offset_1 + lightspot.shape[0] + face_location[1]
            x1_1, x2_1 = x_offset_1 + face_location[0], x_offset_1 + lightspot.shape[1] + face_location[0]

            #print("X1_1 : Y1_1 : X2_1 : Y2_1")
            #print(x1_1, y1_1, x2_1, y2_1)

            # Adjust alpha levels to extract transparency
            alpha_s = lightspot[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            
            image = cv2.imread(imagesFLR) 

            # Apply light spot to cropped image of face 
            for c in range(0, 3):
                image[y1_1:y2_1, x1_1:x2_1, c] = (alpha_s * lightspot[:, :, c] + alpha_l * image[y1_1:y2_1, x1_1:x2_1, c])

            image, confidence = faceDetection(image, net) # Detect face 

            numberOfInts = numberOfInts + 1
            # Confidence score and image 
            confidence_image = []
            confidence_image.append(confidence)
            confidence_image.append(image)

            imageList.append(confidence_image) 

    # Pull the scores out of the list 
    confidenceNumber = []
    for i in range(0, len(imageList)):
        confidenceNumber.append(imageList[i][0])

    # Find the smallest value index 
    lowestConfidenceIndex = confidenceNumber.index(min(confidenceNumber))
    
    stats = []

    stats.append(numberOfInts)
    stats.append(lowestConfidenceIndex)
    stats.append(confidenceNumber[lowestConfidenceIndex])
    
    #print("FACE LOCATION")
    #print(face_location)
    #print("IMAGE DIM")
    #print(image_dimensions)
    #print("Number of Interations:")
    #print(numberOfInts)
    #print("Lowest Index:")
    #print(lowestConfidenceIndex)
    #print("Lowest Score:")
    #print(confidenceNumber[lowestConfidenceIndex])

    return imageList[lowestConfidenceIndex][1], stats



# Find the inital Confidence 
def initalConfidence(imagesFLR, net):
    print("Determining Image Confidence... [initalConfidence]")
    
    boxFLR = []
    finImagesFLR = []
    image_dimensionsFLR = []

    # Loop thru each photo
    for imagePerspective in range(0, len(imagesFLR)):
        # load the input image and construct an input blob for the image
        # by resizing to a fixed 300x300 pixels and then normalizing it
        image = cv2.imread(imagesFLR[imagePerspective]) 
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        # pass the blob through the network and obtain the detections and
        # predictions
        #print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
        
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                boxFLR.append(box.astype("int")) # Convert to int 
                #print("Detection Location: ")
                #print(detections[0, 0, i, 3:7])
                #print("Image W H W H: ")
                # Store image size 
                image_dimensionsFLR.append([w,h,w,h])
                #print([w, h, w, h])
                #print("Box: ")
                #print(boxFLR[imagePerspective])
                (startX, startY, endX, endY) = boxFLR[imagePerspective]
            
                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                finImagesFLR.append(image)
    
    return boxFLR, finImagesFLR, image_dimensionsFLR
    

# Find the inital Confidence 
def faceDetection(image, net):
    #print(" ------------------------------------------------------ Face Detection (Func)")
    
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # pass the blob through the network and obtain the detections and
    # predictions
    #print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    # Store List of the confidence scores. There could be more than one in an image
    confidenceScoreList = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence (0.5 default)
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the
            confidenceScore = (confidence*100)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            #print("Detection Location: ")
            #print(detections[0, 0, i, 3:7])
            # Store image size 
            (startX, startY, endX, endY) = box.astype("int")
            # Add to list
            confidenceScoreList.append(confidenceScore) #
            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    

    return image, confidenceScoreList


def main():

    # Timer 
    start = time.time()

    # Store image input 
    imagesFLR = [args["imageFront"], args["imageLeft"], args["imageRight"]]
    
    # load our serialized model from disk
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    face_locationFLR, imageWithConfidenceFLR, image_dimensionsFLR = initalConfidence(imagesFLR, net) # Get the inital images, face locations, original images dimensions 

    # Save original FD images
    cv2.imwrite("./output_images/front_Inital_FD.jpg", imageWithConfidenceFLR[0])
    cv2.imwrite("./output_images/left_Inital_FD.jpg", imageWithConfidenceFLR[1])
    cv2.imwrite("./output_images/right_Inital_FD.jpg", imageWithConfidenceFLR[2])
    print('Saved inital OpenCV FD: ./output_images/front_Inital_FD.jpg')
    print('Saved inital OpenCV FD: ./output_images/left_Inital_FD.jpg')
    print('Saved inital OpenCV FD: ./output_images/right_Inital_FD.jpg')

    crop_imgFLR = cropImage(face_locationFLR, imagesFLR) # Crop the images

    lightspotFLR = lightSpotModel(image_dimensionsFLR, face_locationFLR, imagesFLR) # Apply light spot to image. Cropped image, original image dimensions, face dimensions  

    attackImages = [] # Store the final images for attack 
    
    pixelIncrement = args["pixIncr"] # Pixel Movement Range, the amount to move the lightspot on the face each iteration 
    
    lpoFrontImage, statsFront = LPO_Front(crop_imgFLR[0], lightspotFLR[0], face_locationFLR[0], image_dimensionsFLR[0], pixelIncrement, net)
    attackImages.append(lpoFrontImage) # Apply the optimiser for 3 light spots 

    lpoLeftImage, statsLeft = LPO_LR(crop_imgFLR[1], lightspotFLR[1], face_locationFLR[1], image_dimensionsFLR[1], imagesFLR[1], pixelIncrement, net) # Apply the optimiser for only 1 light spot
    attackImages.append(lpoLeftImage)

    lpoRightImage, statsRight = LPO_LR(crop_imgFLR[2], lightspotFLR[2], face_locationFLR[2], image_dimensionsFLR[2], imagesFLR[2], pixelIncrement, net) # Apply the optimiser for only 1 light spo
    attackImages.append(lpoRightImage)

    # Save LPO images 
    cv2.imwrite("./output_images/front_LPO.jpg", attackImages[0])
    cv2.imwrite("./output_images/left_LPO.jpg", attackImages[1])
    cv2.imwrite("./output_images/right_LPO.jpg", attackImages[2])
    print('Saved LPO Suggestion: ./output_images/front_LPO.jpg')
    print('Saved LPO Suggestion: ./output_images/left_LPO.jpg')
    print('Saved LPO Suggestion: ./output_images/right_LPO.jpg')

    end = time.time()

    print("-----------------------------------------------------")
    print("Images have been saved the to output_images folder")
    print("-----------------------------------------------------")
    print("Please adjust hat to fit recommended lightspots. Take a photo and save it somewhere.")
    print("-----------------------------------------------------")

    # Run on adjusted image 
    adjusted_front_path = input("Type path to Adjusted FRONT Image here: ")
    print("-----------------------------------------------------")
    adjusted_left_path = input("Type path to Adjusted LEFT Image here: ")
    print("-----------------------------------------------------")
    adjusted_right_path = input("Type path to Adjusted RIGHT Image here: ")

    # Run testing again 
    imagesAdjusted = [adjusted_front_path, adjusted_left_path, adjusted_right_path]
    face_location_Adjus, imageWithConfidenceFLR_Adjus, image_dimensionsFLR_Adjus = initalConfidence(imagesAdjusted, net) # Grab the new confidence of the retaken photos 

    # Save LPO images 
    cv2.imwrite("./output_images/front_Final_FD.jpg", imageWithConfidenceFLR_Adjus[0])
    cv2.imwrite("./output_images/left_Final_FD.jpg", imageWithConfidenceFLR_Adjus[1])
    cv2.imwrite("./output_images/right_Final_FD.jpg", imageWithConfidenceFLR_Adjus[2])
    print('Saved Final OpenCV FD: ./output_images/front_Final_FD.jpg')
    print('Saved Final OpenCV FD: ./output_images/left_Final_FD.jpg')
    print('Saved Final OpenCV FD: ./output_images/right_Final_FD.jpg')
    
    print("-----------------------------------------------------")
    print("The Statistics")
    print("-----------------------------------------------------")
    print("Time Taken (LPO): {}".format((end - start)))
    print("Pixel Increament Size: {}".format(args["pixIncr"]))
    print("Front Inital Image: ")
    print(" - Image Size: {}".format(image_dimensionsFLR[0]))
    print(" - Face Location: {}".format(face_locationFLR[0]))
    print(" - Lightspot rescaled to: [{},{}]".format(lightspotFLR[0].shape[1], lightspotFLR[0].shape[0]))
    print(" - Number of iterations on LPO Front: {}".format(statsFront[0]))
    print(" - Lowest Confidence Index LPO Front: {}".format(statsFront[1]))
    print(" - Lowest Confidence Value LPO Front: {}".format(statsFront[2]))
    print("Left Inital Image: ")
    print(" - Image Size: {}".format(image_dimensionsFLR[1]))
    print(" - Face Location: {}".format(face_locationFLR[1]))
    print(" - Lightspot rescaled to: [{},{}]".format(lightspotFLR[1].shape[1], lightspotFLR[1].shape[0]))
    print(" - Number of iterations on LPO Left: {}".format(statsLeft[0]))
    print(" - Lowest Confidence Index LPO Left: {}".format(statsLeft[1]))
    print(" - Lowest Confidence Value LPO Left: {}".format(statsLeft[2]))
    print("Right Inital Image: ")
    print(" - Image Size: {}".format(image_dimensionsFLR[2]))
    print(" - Face Location: {}".format(face_locationFLR[2]))
    print(" - Lightspot rescaled to: [{},{}]".format(lightspotFLR[2].shape[1], lightspotFLR[2].shape[0]))
    print(" - Number of iterations on LPO Right: {}".format(statsRight[0]))
    print(" - Lowest Confidence Index LPO Right: {}".format(statsRight[1]))
    print(" - Lowest Confidence Value LPO Right: {}".format(statsRight[2]))

if __name__ == "__main__":
    main()