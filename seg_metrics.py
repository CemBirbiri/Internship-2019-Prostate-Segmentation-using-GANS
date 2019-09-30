from binary import dc, jc, hd, hd95
import matplotlib.pyplot as plt 
import pydicom
import numpy as np 
import cv2 
from functions_compare import *

fichier_dice = open("/home/azh2/Desktop/CycleGAN-master/results/raw+sp1200+noisy+blurred5/latest_test/images/Dice.txt","w")

#fichier_haus = open("/home/azh2/Desktop/CycleGAN-master/result_txts/Hausdorff.txt","w")
fichier_haus95 = open("/home/azh2/Desktop/CycleGAN-master/results/raw+sp1200+noisy+blurred5/latest_test/images/Hausdorff95.txt","w")

print('opening successful')

fichier_dice.write("Case;Number;Raw+noisy\n")


fichier_haus95.write("Case;Number;Raw+noisy\n")

print('write sucessful')

nb = str(50)
mod = 'adc'

for a in range (16,26) : 
    
    if a == 16 : 
        x1 = 24
        x2 = 16
    if a == 17 : 
        x1 = 25
        x2 = 16
    if a == 18 : 
        x1 = 26
        x2 = 16
    if a == 19 : 
        x1 = 27
        x2 = 32
    if a == 20 : 
        x1 = 29
        x2 = 32
    if a == 21 : 
        x1 = 3
        x2 = 16
    if a == 22 : 
        x1 = 30
        x2 = 32
    if a == 23 :   
        x1 = 31
        x2 = 32
    if a == 24 :  
        x1 = 32
        x2 = 32
    if a == 25 :   
        x1 = 33
        x2 = 16

    for i in range (1,x2+1):
        im = 'rST0000'+str(a)+' ('+str(x1)+')_%d' %i

        print(im)

        #path_mask_ori = mod+'/prostate_'+mod+'_raw_lr_low/'+nb+'_net_G_val/images/target/r'+im+'.png'
        path_mask_ori = '/home/azh2/Desktop/CycleGAN-master/results/raw+sp1200+noisy+blurred5/latest_test/images/real_B/'+im+'.png'

        #path_mask_raw = mod+'/prostate_'+mod+'_raw_lr_low/'+nb+'_net_G_val/images/post/r'+im+'.png'
        #path_mask_raw ='/home/azh2/Desktop/CycleGAN-master/results/no_black_nrc1/latest_test/images/post/'+im+'.png'
        
        #path_mask_raw_noisy = mod+'/prostate_'+mod+'_raw_noisy_lr_low/'+nb+'_net_G_val/images/post/r'+im+'.png'
        path_mask_raw_noisy = '/home/azh2/Desktop/CycleGAN-master/results/raw+sp1200+noisy+blurred5/latest_test/images/post/'+im+'.png'
        
        

        fichier_dice.write(str(a)+';'+str(i)+';')
        
        #fichier_haus.write(str(a)+';'+str(i)+';')
        fichier_haus95.write(str(a)+';'+str(i)+';')
        
        print(a,i)


        for k in range (0,1):
            
            if k == 0 :
                path_mask = path_mask_raw_noisy
            
            if k == 1 :
                path_mask = path_mask_raw 
           
     
               

            mask_reference_gray = cv2.imread(path_mask_ori,0)
            mask_result_gray = cv2.imread(path_mask,0)

            mask_reference = gray2binary(mask_reference_gray)
            mask_result = gray2binary(mask_result_gray)

            #Dice distance 
            dice = dc(mask_result, mask_reference)

            #Haudorff distance 
            haus = hd(mask_result, mask_reference, voxelspacing=1.0156, connectivity=1)

            #Hausdorff distance 95 
            haus2 = hd95(mask_result, mask_reference, voxelspacing=1.0156, connectivity=1)

            fichier_dice.write(str(dice)+';')
            
            #fichier_haus.write(str(haus)+';')
            fichier_haus95.write(str(haus2)+';')

        fichier_dice.write("\n")
       
        #fichier_haus.write("\n")
        fichier_haus95.write("\n")

		
fichier_dice.close()

#fichier_haus.close()
fichier_haus95.close()
