import matplotlib.pyplot as plt 
import pydicom
import numpy as np 
import cv2 
from skimage.segmentation import clear_border 
from skimage.segmentation import mark_boundaries
from skimage.measure import label
from skimage.color import label2rgb 
from functions_compare import *

fichier2 = open("/home/azh2/Desktop/CycleGAN-master/results/raw+sp1200+noisy+blurred5/latest_test/images/det_metrics_adc.txt","w")

#fichier2.write("Case;Number;Raw TP;FP;FN;TN;TN;Raw+noisy TP;FP;FN;TN;\n")
fichier2.write("Case;Number;Raw+noisy TP;FP;FN;TN;\n")

nb = str(50)
red = (0,0,255)
blue = (255,0,0)
green = (0,255,0)
font = cv2.FONT_HERSHEY_SIMPLEX


FP_raw = 0 
FP_raw_noisy = 0

TP_raw = 0 
TP_raw_noisy = 0

FN_raw = 0 
FN_raw_noisy = 0

TN_raw = 0 
TN_raw_noisy = 0

for a in range(16,26):
    if a==16:
        x1=24
        x2=16
        
    
    if a==17:
        x1=25
        x2=16
           
    
    if a==18:
        x1=26
        x2=16
        
    
    if a==19:
        x1=27
        x2=32
        
    
    if a==20:
        x1=29
        x2=32
        
    
    if a==21:
        x1=3
        x2=16
        
    
    if a==22:
        x1=30
        x2=32
        
    
    if a==23:
        x1=31
        x2=32
        
    
    if a==24:
        x1=32
        x2=32
        
    
    if a==25:
        x1=33
        x2=16
        
    for i in range (1,x2+1):
        #im = 'rST0000'+str(a)+' ('+str(x1)+')_%d' %i
        
        im1= 'rST0000'+str(a)+' ('+str(x1)+')_%d' %i

        print(im1)

        

        #path_raw='/home/azh2/Desktop/CycleGAN-master/results/no_black_nrc1/latest_test/images/real_A/'+im1+'.png'#input
       
       

        path_mask_ori='/home/azh2/Desktop/CycleGAN-master/results/raw+sp1200+noisy+blurred5/latest_test/images/real_B/'+im1+'.png'#target
        #path_mask_raw = '/home/azh2/Desktop/CycleGAN-master/results/n1/latest_test/images/real_A/'+im+'.png' #post
      
        path_mask_raw_noisy='/home/azh2/Desktop/CycleGAN-master/results/raw+sp1200+noisy+blurred5/latest_test/images/post/'+im1+'.png'#post
        

 

        mask_ori_gray = cv2.imread(path_mask_ori,0)
        #mask_raw_gray = cv2.imread(path_mask_raw,0)
        mask_raw_noisy_gray =cv2.imread(path_mask_raw_noisy,0)
       

        mask_ori_bin =gray2binary (mask_ori_gray)
        #mask_raw_bin =gray2binary (mask_raw_gray)
        mask_raw_noisy_bin =gray2binary (mask_raw_noisy_gray)
        

        ori_dice= np.reshape(mask_ori_bin,(-1))
        #raw_dice = np.reshape(mask_raw_bin,(-1))
        raw_noisy_dice = np.reshape(mask_raw_noisy_bin,(-1))
        

        
        if ori_dice.sum()==0 and raw_noisy_dice.sum()==0:

            dist_dice_raw_noisy_c=0.5
        #dist_dice_raw_c = distance.dice (ori_dice,raw_dice)
        else:
            dist_dice_raw_noisy_c = distance.dice (ori_dice,raw_noisy_dice) 
        
        

        contours_ori,hierarchy1=cv2.findContours(mask_ori_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        
        ##### TP,FP,TN,FN -------------------------------------------
 


        #FP and TN 

        if (len(contours_ori)== 0 ) :  #original image is fully black

            #if dist_dice_raw_c == 1 :
                #FP_raw += 1
            
            if dist_dice_raw_noisy_c == 1 : #ayrik kumeler
                FP_raw_noisy+=1
                
            
            #if dist_dice_raw_c != 1 :
                #TN_raw += 1
            
            if dist_dice_raw_noisy_c != 1: #kesisim var
                TN_raw_noisy+=1 #here
                print("bebek")
            

        if (len(contours_ori)!= 0 ) : #original image is NOT fully black
            #if dist_dice_raw_c == 1 :
               #FN_raw += 1
            
            if dist_dice_raw_noisy_c == 1 :  #ayrik kumeler
               FN_raw_noisy+=1
            

            
            #if dist_dice_raw_c != 1 :
                #TP_raw += 1
            
            if dist_dice_raw_noisy_c != 1 : #kesisim var
                TP_raw_noisy+=1
            


        fichier2.write(str(a)+';')
        fichier2.write(str(i)+';')
        '''
        fichier2.write(str(TP_raw)+';')
        fichier2.write(str(FP_raw)+';')
        fichier2.write(str(FN_raw)+';')
        fichier2.write(str(TN_raw)+';')'''
     
        fichier2.write(str(TP_raw_noisy)+';')
        fichier2.write(str(FP_raw_noisy)+';')
        fichier2.write(str(FN_raw_noisy)+';')
        fichier2.write(str(TN_raw_noisy)+';')
     
        fichier2.write("\n")

        ##--------------------------------------------------

        
fichier2.write("end \n")
fichier2.close()