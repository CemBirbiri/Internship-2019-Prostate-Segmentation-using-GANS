import matplotlib.pyplot as plt 
import pydicom
import numpy as np 
import cv2 
from skimage.segmentation import clear_border 
from skimage.segmentation import mark_boundaries
from skimage.measure import label
from skimage.color import label2rgb 
from functions_compare import *

fichier = open("/home/azh2/Desktop/results.txt","w")
fichier2 = open("/home/azh2/Desktop/results2.txt","w")

fichier.write("Case;Number;Dice raw;Dice raw+noisy;Haus raw;Haus raw+noisy\n")
fichier2.write("Case;Number;Raw TP;FP;FN;TN;Raw+noisy TP;FP;FN;TN\n")


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


for a in range(16,17):
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
        im = 'rST0000'+str(a)+' ('+str(x1)+')_%d' %i
        im1= 'nST0000'+str(a)+' ('+str(x1)+')_%d' %i

        

        path_raw = '/home/azh2/Desktop/CycleGAN-master/results/expt1/latest_test/images/real_A/'+im+'.png' #realA
        #path_mask_ori = '/home/azh2/Desktop/CycleGAN-master/results/expt1/latest_test/images/real_B/'+im+'.png' #realB
        #path_mask_raw = 'adc/prostate_adc_raw_lr_low/'+nb+'_net_G_val/images/post/r'+im+'.png'
        
        #path_mask_raw_noisy = '/home/azh2/Desktop/CycleGAN-master/results/expt1/latest_test/images/post/'+im1+'.png'
        path_mask_ori ='/media/azh2/Elements/Amelie_prostateSegmentation/Segmentation/Results/adc/prostate_adc_raw_lr_low/50_net_G_val/images/target/'+im+'.png'


        path_mask_raw ='/media/azh2/Elements/Amelie_prostateSegmentation/Segmentation/Results/adc/prostate_adc_raw_lr_low/50_net_G_val/images/post/'+im+'.png'
        path_mask_raw_noisy ='/media/azh2/Elements/Amelie_prostateSegmentation/Segmentation/Results/adc/prostate_adc_raw_noisy_lr_low/50_net_G_val/images/post/'+im+'.png'

        raw = cv2.imread(path_raw)
        mask_ori = cv2.imread(path_mask_ori)
        #mask_raw = cv2.imread(path_mask_raw) 
        mask_raw_noisy =cv2.imread(path_mask_raw_noisy)
        

        mask_ori_gray = cv2.imread(path_mask_ori,0)
        #mask_raw_gray = cv2.imread(path_mask_raw,0)
        mask_raw_noisy_gray =cv2.imread(path_mask_raw_noisy,0)
        
        ##### Dice distance ---------------------------------------------------------
        mask_ori_bin =gray2binary (mask_ori_gray)
        #mask_raw_bin =gray2binary (mask_raw_gray)
        mask_raw_noisy_bin =gray2binary (mask_raw_noisy_gray)
    

        ori_dice= np.reshape(mask_ori_bin,(-1))
        #raw_dice = np.reshape(mask_raw_bin,(-1)) 
        raw_noisy_dice = np.reshape(mask_raw_noisy_bin,(-1))

        

        
        #dist_dice_raw_c = distance.dice (ori_dice,raw_dice)
        dist_dice_raw_noisy_c = distance.dice (ori_dice,raw_noisy_dice) 
        
        

        #dist_dice_raw = (str(distance.dice (ori_dice,raw_dice)))[:6]
        dist_dice_raw_noisy = (str(distance.dice (ori_dice,raw_noisy_dice) ))[:6]

        print(dist_dice_raw_noisy)

        

        ### end Dice distance ---------------------------------------------

        ### Contours ---------------------------------------------------------

        contours_ori,hierarchy1=cv2.findContours(mask_ori_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #img2,contours_raw,hierarchy2=cv2.findContours(mask_raw_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours_raw_noisy,hierarchy6=cv2.findContours(mask_raw_noisy_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
       

        ### end Contours ----------------------------

        ## max Hausdorff distance -------------
       
        #hausdorff_max_raw = (str(redifined_distance_hausdorff (contours_ori, contours_raw, maximum=True)))[:6]
        hausdorff_max_raw_noisy = (str(redifined_distance_hausdorff (contours_ori, contours_raw_noisy, maximum=True)))[:6]
       
        ## end max hausdorff distance ----------------------

 
        ##### TP,FP,TN,FN -------------------------------------------
 


        #FP and TN 

        if (len(contours_ori)== 0 ) : 

            #if dist_dice_raw_c == 1 :
                #FP_raw += 1
            if dist_dice_raw_noisy_c == 1 :
                FP_raw_noisy+=1
            

            #if dist_dice_raw_c != 1 :
                #TN_raw += 1
                #print('raw')
                #print(a)
                #print(i)
 
            if dist_dice_raw_noisy_c != 1:
                TN_raw_noisy+=1
                print('raw noisy')
                print(a)
                print(i)
            

        if (len(contours_ori)!= 0 ) : 
            #if dist_dice_raw_c == 1 :
                #FN_raw += 1
           
            if dist_dice_raw_noisy_c == 1 :
                FN_raw_noisy+=1
            
            

            #if dist_dice_raw_c != 1 :
                #TP_raw += 1
            if dist_dice_raw_noisy_c != 1 :
                TP_raw_noisy+=1
            

        fichier2.write(str(a)+';')
        fichier2.write(str(i)+';')
       
        #fichier2.write(str(TP_raw)+';')
        #fichier2.write(str(FP_raw)+';')
        #fichier2.write(str(FN_raw)+';')
        #fichier2.write(str(TN_raw)+';')
      
        fichier2.write(str(TP_raw_noisy)+';')
        fichier2.write(str(FP_raw_noisy)+';')
        fichier2.write(str(FN_raw_noisy)+';')
        fichier2.write(str(TN_raw_noisy)+';')
        
        fichier2.write("\n")

        ##--------------------------------------------------

        
        if i >= m[0] and i <= m[1] :
           
            fichier.write(str(a)+';')
            fichier.write(str(i)+';')
           
            #fichier.write(dist_dice_raw+';')     
            fichier.write(dist_dice_raw_noisy+';')
            
            
            #fichier.write(hausdorff_max_raw+';')         
            fichier.write(hausdorff_max_raw_noisy+';')
            
            fichier.write("\n")


fichier.write("end \n")
fichier.close()

fichier2.write("end \n")
fichier2.close()








