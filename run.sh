#!/bin/sh

# To run type: ./run.sh FRONT_IMAGE_PATH LEFT_IMAGE_PATH RIGHT_IMAGE_PATH PIX_INCR

# Example ./run.sh ./images/front.jpg ./images/left.jpg ./images/right.jpg 100
# chmod 755 run.sh to update the shell script once its changed 

# Run the facenet system to get initial confidence 
cd ./facenet_pytorch_specific
python3 -m inference.classifier --image-path ../$1 --save-dir ../output_images --inital-bool True
python3 -m inference.classifier --image-path ../$2 --save-dir ../output_images --inital-bool True
python3 -m inference.classifier --image-path ../$3 --save-dir ../output_images --inital-bool True

# Move back to LPO system 
cd ../
python3 detect_faces.py --imageFront $1 --imageLeft $2 --imageRight $3 --pixIncr $4 --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# Move back to the FR system to test Adjusted Photos 
cd ./facenet_pytorch_specific
python3 -m inference.classifier --image-path .././images/front_ad.jpg --save-dir ../output_images --inital-bool False
python3 -m inference.classifier --image-path .././images/left_ad.jpg --save-dir ../output_images --inital-bool False
python3 -m inference.classifier --image-path .././images/right_ad.jpg --save-dir ../output_images --inital-bool False