#!/usr/bin/env python

import os
import joblib
import argparse
from PIL import Image
from .util import draw_bb_on_img
from .constants import MODEL_PATH
from face_recognition import preprocessing


def parse_args():
    parser = argparse.ArgumentParser(
        'Script for detecting and classifying faces on user-provided image. This script will process image, draw '
        'bounding boxes and labels on image and display it. It will also optionally save that image.')
    parser.add_argument('--image-path', required=True, help='Path to image file.')
    parser.add_argument('--save-dir', required=True, help='If save dir is provided image will be saved to specified directory.')
    parser.add_argument('--inital-bool', required=True, help='Type True if you are testing on a photo with IRH OFF')
    return parser.parse_args()


def recognise_faces(img):
    faces = joblib.load(MODEL_PATH)(img)
    if faces:
        draw_bb_on_img(faces, img)
    return faces, img


def main():
    args = parse_args()
    preprocess = preprocessing.ExifOrientationNormalize()
    img = Image.open(args.image_path)
    filename = img.filename
    img = preprocess(img)
    img = img.convert('RGB')

    faces, img = recognise_faces(img)
    if not faces:
        print('No faces found in this image.')

    if args.save_dir:
        basename = os.path.basename(filename)
        name = basename.split('.')[0]
        ext = basename.split('.')[1]
        if (args.inital_bool == "True"):
            img.save(args.save_dir + '/{}_Initial_FR.{}'.format(name, ext))
            save_name = args.save_dir + '/{}_Initial_FR.{}'.format(name, ext)
            print('Saved inital FaceNet FR: ' + save_name)
        else: 
            img.save(args.save_dir + '/{}_Final_FR.{}'.format(name, ext))
            save_name = args.save_dir + '/{}_Final_FR.{}'.format(name, ext)
            print('Saved final FaceNet FR: ' + save_name)
        


    #img.show()


if __name__ == '__main__':
    main()
