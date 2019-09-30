import matplotlib.pyplot as plt 
import pydicom
import numpy as np 
import cv2 
from skimage.segmentation import clear_border 
from skimage.segmentation import mark_boundaries
from skimage.measure import label
from skimage.color import label2rgb 
from functions_compare import *


 




def postprocess(a,x1,x2):


    
    for i in range(1,x2+1):


        im = 'rST0000'+str(a)+' ('+str(x1)+')_%d' %i

        print(im)
        
        #path_raw = '/home/azh2/Desktop/CycleGAN-master/results/n1/latest_test/images/real_A/'+im+'.png'
        path_mask_ori = '/home/azh2/Desktop/CycleGAN-master/results/raw+sp1200+noisy+blurred5/latest_test/images/real_B/'+im+'.png'
        path_mask_pred = '/home/azh2/Desktop/CycleGAN-master/results/raw+sp1200+noisy+blurred5/latest_test/images/fake_B/'+im+'.png'

        #raw= cv2.imread(path_raw)
        mask_ori = cv2.imread(path_mask_ori)
        mask_pred = cv2.imread(path_mask_pred)
                


        mask_ori1 = cv2.imread(path_mask_ori,0)
        mask_pred1 = cv2.imread(path_mask_pred,0)
        mask_pred_bin = gray2binary (mask_pred1)
        
        cleared_bin = clear_border (mask_pred_bin)
        labels, num = label(cleared_bin, return_num = True)
        mask_return_bin= keep_biggest_label(labels, num) 
        mask_return_gray = binary2gray (mask_return_bin)
                   
        cv2.imwrite('/home/azh2/Desktop/CycleGAN-master/results/raw+sp1200+noisy+blurred5/latest_test/images/post/'+im+'.png',mask_return_gray)



for a in range(16,26):
    if a==16:
        x1=24
        x2=16
        postprocess(a,x1,x2)
    
    if a==17:
        x1=25
        x2=16
        postprocess(a,x1,x2)    
    
    if a==18:
        x1=26
        x2=16
        postprocess(a,x1,x2)
    
    if a==19:
        x1=27
        x2=32
        postprocess(a,x1,x2)
    
    if a==20:
        x1=29
        x2=32
        postprocess(a,x1,x2)
    
    if a==21:
        x1=3
        x2=16
        postprocess(a,x1,x2)
    
    if a==22:
        x1=30
        x2=32
        postprocess(a,x1,x2)
    
    if a==23:
        x1=31
        x2=32
        postprocess(a,x1,x2)
    
    if a==24:
        x1=32
        x2=32
        postprocess(a,x1,x2)
    
    if a==25:
        x1=33
        x2=16
        postprocess(a,x1,x2)


