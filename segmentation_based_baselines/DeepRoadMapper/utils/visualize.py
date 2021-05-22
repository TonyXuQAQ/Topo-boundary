from PIL import Image, ImageDraw
import json
import numpy as np
import os
import cv2

def visualize():
    image_list = os.listdir('./records/reason/test/vis')
    for ii,image_name in enumerate(image_list):
        image = Image.fromarray(np.array(Image.open(os.path.join('./dataset/cropped_tiff',image_name[:-3]+'tiff')))[:,:,:3])
        p = image.load()
        im = cv2.imread(os.path.join('./records/reason/test/vis',image_name))
        kernel = np.ones((7,7), np.uint8)
        dilate = cv2.dilate(im, kernel)

        foreground_pixels = np.where(dilate[:,:,-1]!=0)
        foreground_pixels = [[foreground_pixels[1][x],foreground_pixels[0][x]] for x in range(len(foreground_pixels[0]))]
        for point in foreground_pixels:
            p[int(point[0]),int(point[1])] = (153, 255, 51)
        
        image.save(os.path.join('./records/reason/test/final_vis',image_name))
        print('Visualizing',ii,'/',len(image_list))

def copy_gt_image():
    for img in os.listdir('./img'):
        gt = Image.fromarray(np.array(Image.open(os.path.join('./dataset/cropped_tiff',img[:-3]+'tiff')))[:,:,:3])
        p = gt.load()
        im = cv2.imread(os.path.join('./dataset/labels/binary_map',img))
        kernel = np.ones((7,7), np.uint8)
        dilate = cv2.dilate(im, kernel)

        foreground_pixels = np.where(dilate[:,:,0]!=0)
        foreground_pixels = [[foreground_pixels[1][x],foreground_pixels[0][x]] for x in range(len(foreground_pixels[0]))]
        for point in foreground_pixels:
            p[int(point[0]),int(point[1])] = (0,255,255)
        
        gt.save(os.path.join('./img','gt_{}'.format(img)))
    

def main():
    visualize()
    # copy_gt_image()

if __name__=='__main__':
    main()