# Prostate-Segmentation-using-GANS
2019 Summer internship at Aberystwyth University,UK under the supervision of Reyer Zwiggelaar.


Generative Adversarial Networks(GANs) have shown great success in generating different kinds of visual content. 
GANs have also being used in medical image applications such as medical image reconstruction, segmentation, detection,  
image synthesis or classification. In addition, with their ability to produce highly realistic images, GANs are very 
promising where the absence of labeled data in medical field and in medical image analysis problems such as detection , 
classification or segmentation.  In this paper we demonstrated the performance of the conditional GAN(cGAN) and cycleGAN 
networks on common prostate MR images(ADC,DWI and T2W) in terms of detection and segmentation of prostate.
To the best of our knowledge, this is the first usage of cycleGAN performed on prostate MR image scans. 
Furthermore, we proposed three different data augmentation methods and investigated the robustness of cGAN and 
cycleGAN models against superpixelizing , adding Gaussian noise and smoothing the training data. 
First we observed the performance of U-Net,cGAN and cycleGAN networks on raw prostate images then performance of 
cGAN and cycleGAN networks on datasets which are created with our data augmentation methods. Based on the detection 
and segmentation metrics, all the three data augmentation methods improved the performance in each 3D prostate modality.





These files include detection and segmentation metrics that we use and also data augmentation methods 
such as superpixeling, blurring and adding gaussian noise.

main programs are:

postprocessing 
    do the post processing step and put images in the folder 'post' ; erase everything near the border and 
keep the biggest label
    postprocessing.py
    

compare 
    compare results from raw/noisy/denoised/raw+noisy/raw+denoised/raw+noisy+denoised 
    in image : big image with mask predicted, hausdorff distance printed, dice coeff        
    compare.py
    

det_metrics 
    TP,FP,TN,FN for all the datasets in the file results.txt
    det_metrics.py
    

seg_metrics 
    dice, haus 95 for each dataset im different files 
    seg_metrics.py
   

Data_Augmentation.py
    
