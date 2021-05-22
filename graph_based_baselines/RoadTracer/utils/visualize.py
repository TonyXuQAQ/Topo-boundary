from PIL import Image, ImageDraw
import json
import numpy as np
import os
import cv2

def visualize():
    image_list = os.listdir('./records/test/skeleton')
    for ii,image_name in enumerate(image_list):
        image = Image.fromarray(np.array(Image.open(os.path.join('./dataset/cropped_tiff',image_name[:-3]+'tiff')))[:,:,:3])
        p = image.load()
        im = cv2.imread(os.path.join('./records/test/skeleton',image_name))
        kernel = np.ones((5,5), np.uint8)
        dilate = cv2.dilate(im, kernel)

        foreground_pixels = np.where(dilate[:,:,0]!=0)
        foreground_pixels = [[foreground_pixels[1][x],foreground_pixels[0][x]] for x in range(len(foreground_pixels[0]))]
        for point in foreground_pixels:
            p[int(point[0]),int(point[1])] = (255,165,0)
        
        draw = ImageDraw.Draw(image)
        with open('./records/test/vertices_record/{}'.format(image_name[:-3]+'json'),'r') as jf:
            points = json.load(jf)
        for point in points:
            draw.ellipse((point[1]-4,point[0]-4,point[1]+4,point[0]+4),fill='yellow',outline='yellow')
        image.save(os.path.join('./records/test/final_vis',image_name))
        print('Visualizing',ii,'/',len(image_list))

def main():
    visualize()

if __name__=='__main__':
    main()