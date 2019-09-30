from skimage.segmentation import slic
import cv2
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt

import skimage.color as color
import numpy as np
import matplotlib.pyplot as plt
import skimage.data as data

#import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from skimage.segmentation import mark_boundaries

import skimage.segmentation as seg

from scipy import ndimage, misc





from skimage.segmentation import slic
import cv2

import matplotlib.pyplot as plt

import skimage.color as color
import numpy as np
import matplotlib.pyplot as plt
import skimage.data as data

#import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from skimage.segmentation import mark_boundaries

import skimage.segmentation as seg


def superpixel(orig):
	def segment_colorfulness(image, mask):
		# split the image into its respective RGB components, then mask
		# each of the individual RGB channels so we can compute
		# statistics only for the masked region
		(B, G, R) = cv2.split(image.astype("float"))
		R = np.ma.masked_array(R, mask=mask)

		G = np.ma.masked_array(B, mask=mask)
		B = np.ma.masked_array(B, mask=mask)


		return (R.sum()+G.sum()+B.sum())/(128*256)

	# load the image in OpenCV format so we can draw on it later, then
	# allocate memory for the superpixel colorfulness visualization
	#orig = cv2.imread(args["image"])
	vis = np.zeros(orig.shape[:2], dtype="float")
	 
	# load the image and apply SLIC superpixel segmentation to it via
	# scikit-image
	#image = io.imread(args["image"])
	image=orig
	segments = slic(img_as_float(image), n_segments=1200,
		slic_zero=True)


	#print(orig.shape)


	# loop over each of the unique superpixels
	for v in np.unique(segments):
		# construct a mask for the segment so we can compute image
		# statistics for *only* the masked region
		mask = np.ones(image.shape[:2])
		mask[segments == v] = 0
	 
		# compute the superpixel colorfulness, then update the
		# visualization array
		C = segment_colorfulness(orig, mask)
		vis[segments == v] = C


	# scale the visualization image from an unrestricted floating point
	# to unsigned 8-bit integer array so we can use it with OpenCV and
	# display it to our screen
	vis = rescale_intensity(vis, out_range=(0, 255)).astype("uint8")
	 
	# overlay the superpixel colorfulness visualization on the original
	# image
	alpha = 0.6
	overlay = np.dstack([vis] * 3)
	output = orig.copy()
	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)


	return output
#*************************************************************************************************************************************************

def crop_image(image):

	img1= image[64:192,:] #first cropping-> make image 128x256x3
	return img1




def is_full_black(image):
	black=0
	white=0
	for i in range(0,256):
		for j in range(0,256):
			for m in range(0,3):
				if image[i][j][m] ==0:
					black=black+1
				else:
					white=white+1
	if white<2400:
		return 1
	else:
		return 0

#**************************************************************************************************************************************************
def separate_images_traininggg(a,x11,x12,x21,x22):



	if a in numbers:
		for i in range(1,x21+1):


			im = 'rST0000'+'0'+str(a)+' ('+str(x11)+')_%d' %i

			print(im)
			path_read = '/media/azh2/Elements/Amelie_prostateSegmentation/Segmentation/datasets/T2W_prostate/raw/train/'+im+'.png'
			img = cv2.imread(path_read,1)

			

			img1= img[:,0:256]
			img2=img[:,256:512]

			if is_full_black(img2)==0:
				img2=crop_image(img2)
				img1=crop_image(img1)

				'''
				#noisy -trainA
				path_noisy='/home/azh2/Desktop/CycleGAN-master/prostate_data/dwi/noisy/trainA/'
				cv2.imwrite(path_noisy+'noisy_'+im+'.png',img1)
				#noisy -trainB
				path_noisy2='/home/azh2/Desktop/CycleGAN-master/prostate_data/dwi/noisy/trainB/'
				cv2.imwrite(path_noisy2+'noisy_'+im+'.png',img2)
				'''
				
				#raw
				path_write1='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/raw/trainA/'
				cv2.imwrite(path_write1+im+'.png',img1)

				#mask
				path_write2='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/raw/trainB/'
				cv2.imwrite(path_write2+im+'.png',img2)
				
				#blurr
				result = ndimage.uniform_filter(img1, size=5, mode='mirror')
				path_write5='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/blurred5/trainA/'
				cv2.imwrite(path_write5+'blurred5_'+im+'.png',result)
				path_blurred_mask ='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/blurred5/trainB/'
				cv2.imwrite(path_blurred_mask+'blurred5_'+im+'.png',img2)


				
				#superpixel-trainA
				super_img = superpixel(img1)
				path_super='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/sp1200/trainA/'
				cv2.imwrite(path_super+'sp1200_'+im+'.png',super_img)
				
				#superpixel-trainB
				path_raw1='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/sp1200/trainB/'
				cv2.imwrite(path_raw1+'sp1200_'+im+'.png',img2)
				

				
			
			
		for j in range(1,x22+1):
			
			im = 'rST0000'+'0'+str(a)+' ('+str(x12)+')_%d' %j
			
			print(im)
			path_read = '/media/azh2/Elements/Amelie_prostateSegmentation/Segmentation/datasets/T2W_prostate/raw/train/'+im+'.png'
			img = cv2.imread(path_read,1)

			

			img1= img[:,0:256]
			img2=img[:,256:512]

			if is_full_black(img2)==0:
				img2=crop_image(img2)
				img1=crop_image(img1)

				'''
				#noisy -trainA
				path_noisy='/home/azh2/Desktop/CycleGAN-master/prostate_data/dwi/noisy/trainA/'
				cv2.imwrite(path_noisy+'noisy_'+im+'.png',img1)
				#noisy -trainB
				path_noisy2='/home/azh2/Desktop/CycleGAN-master/prostate_data/dwi/noisy/trainB/'
				cv2.imwrite(path_noisy2+'noisy_'+im+'.png',img2)
				'''
				
				#raw
				path_write1='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/raw/trainA/'
				cv2.imwrite(path_write1+im+'.png',img1)

				#mask
				path_write2='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/raw/trainB/'
				cv2.imwrite(path_write2+im+'.png',img2)
				
				#blurr
				result = ndimage.uniform_filter(img1, size=5, mode='mirror')
				path_write5='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/blurred5/trainA/'
				cv2.imwrite(path_write5+'blurred5_'+im+'.png',result)
				path_blurred_mask ='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/blurred5/trainB/'
				cv2.imwrite(path_blurred_mask+'blurred5_'+im+'.png',img2)


				
				#superpixel-trainA
				super_img = superpixel(img1)
				path_super='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/sp1200/trainA/'
				cv2.imwrite(path_super+'sp1200_'+im+'.png',super_img)
				
				#superpixel-trainB
				path_raw1='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/sp1200/trainB/'
				cv2.imwrite(path_raw1+'sp1200_'+im+'.png',img2)
				

				
	else:
		for i in range(1,x21+1):


			im = 'rST0000'+str(a)+' ('+str(x11)+')_%d' %i
			
			print(im)
			path_read = '/media/azh2/Elements/Amelie_prostateSegmentation/Segmentation/datasets/T2W_prostate/raw/train/'+im+'.png'
			img = cv2.imread(path_read,1)

			

			img1= img[:,0:256]
			img2=img[:,256:512]

			if is_full_black(img2)==0:
				img2=crop_image(img2)
				img1=crop_image(img1)

				'''
				#noisy -trainA
				path_noisy='/home/azh2/Desktop/CycleGAN-master/prostate_data/dwi/noisy/trainA/'
				cv2.imwrite(path_noisy+'noisy_'+im+'.png',img1)
				#noisy -trainB
				path_noisy2='/home/azh2/Desktop/CycleGAN-master/prostate_data/dwi/noisy/trainB/'
				cv2.imwrite(path_noisy2+'noisy_'+im+'.png',img2)
				'''
				
				#raw
				path_write1='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/raw/trainA/'
				cv2.imwrite(path_write1+im+'.png',img1)

				#mask
				path_write2='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/raw/trainB/'
				cv2.imwrite(path_write2+im+'.png',img2)
				
				#blurr
				result = ndimage.uniform_filter(img1, size=5, mode='mirror')
				path_write5='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/blurred5/trainA/'
				cv2.imwrite(path_write5+'blurred5_'+im+'.png',result)
				path_blurred_mask ='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/blurred5/trainB/'
				cv2.imwrite(path_blurred_mask+'blurred5_'+im+'.png',img2)


				
				#superpixel-trainA
				super_img = superpixel(img1)
				path_super='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/sp1200/trainA/'
				cv2.imwrite(path_super+'sp1200_'+im+'.png',super_img)
				
				#superpixel-trainB
				path_raw1='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/sp1200/trainB/'
				cv2.imwrite(path_raw1+'sp1200_'+im+'.png',img2)
				

		



		for j in range(1,x22+1):
			
			im = 'rST0000'+str(a)+' ('+str(x12)+')_%d' %j
			
			print(im)
			path_read = '/media/azh2/Elements/Amelie_prostateSegmentation/Segmentation/datasets/T2W_prostate/raw/train/'+im+'.png'
			img = cv2.imread(path_read,1)

			

			img1= img[:,0:256]
			img2=img[:,256:512]

			if is_full_black(img2)==0:
				img2=crop_image(img2)
				img1=crop_image(img1)

				'''
				#noisy -trainA
				path_noisy='/home/azh2/Desktop/CycleGAN-master/prostate_data/dwi/noisy/trainA/'
				cv2.imwrite(path_noisy+'noisy_'+im+'.png',img1)
				#noisy -trainB
				path_noisy2='/home/azh2/Desktop/CycleGAN-master/prostate_data/dwi/noisy/trainB/'
				cv2.imwrite(path_noisy2+'noisy_'+im+'.png',img2)
				'''
				
				#raw
				path_write1='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/raw/trainA/'
				cv2.imwrite(path_write1+im+'.png',img1)

				#mask
				path_write2='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/raw/trainB/'
				cv2.imwrite(path_write2+im+'.png',img2)
				
				#blurr
				result = ndimage.uniform_filter(img1, size=5, mode='mirror')
				path_write5='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/blurred5/trainA/'
				cv2.imwrite(path_write5+'blurred5_'+im+'.png',result)
				path_blurred_mask ='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/blurred5/trainB/'
				cv2.imwrite(path_blurred_mask+'blurred5_'+im+'.png',img2)


				
				#superpixel-trainA
				super_img = superpixel(img1)
				path_super='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/sp1200/trainA/'
				cv2.imwrite(path_super+'sp1200_'+im+'.png',super_img)
				
				#superpixel-trainB
				path_raw1='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/sp1200/trainB/'
				cv2.imwrite(path_raw1+'sp1200_'+im+'.png',img2)

				
	

def separate_images_test(a,x1,x2):


	
	for i in range(1,x2+1):


		im = 'rST0000'+str(a)+' ('+str(x1)+')_%d' %i
		

		print(im)
		path_read = '/media/azh2/Elements/Amelie_prostateSegmentation/Segmentation/datasets/T2W_prostate/raw/val/'+im+'.png'
		img = cv2.imread(path_read,1)

			

		img1= img[:,0:256]
		img2=img[:,256:512]

			
		img2=crop_image(img2)
		img1=crop_image(img1)

			

		#raw
		path_write1='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/raw/testA/'
		cv2.imwrite(path_write1+im+'.png',img1)

		#mask
		path_write2='/home/azh2/Desktop/CycleGAN-master/prostate_data/t2w/raw/testB/'
		cv2.imwrite(path_write2+im+'.png',img2)
				
				





#**************************************************************************************************************************************************

def change_image_names(a,x11,x12,x21,x22):
	if a in numbers:
		for i in range(1,x21+1):


			im = 'rST0000'+'0'+str(a)+' ('+str(x11)+')_%d' %i
			print(im)
			path_read = '/home/azh2/Desktop/CycleGAN-master/prostate_data/adc2/raw/trainB/'+im+'.png'
			
			img = cv2.imread(path_read,1)

			path_write3='/home/azh2/Desktop/CycleGAN-master/prostate_data/adc2/raw1'
			cv2.imwrite(path_write3+'1_'+im+'.png',img)


			
			
		for j in range(1,x22+1):
			
			im = 'rST0000'+'0'+str(a)+' ('+str(x12)+')_%d' %j
			print(im)
			path_read = '/home/azh2/Desktop/CycleGAN-master/prostate_data/adc2/raw/trainB/'+im+'.png'
			
			img = cv2.imread(path_read,1)
			

			path_write3='/home/azh2/Desktop/CycleGAN-master/prostate_data/adc2/raw1'
			cv2.imwrite(path_write3+'1_'+im+'.png',img)

			
	else:
		for i in range(1,x21+1):


			im = 'rST0000'+str(a)+' ('+str(x11)+')_%d' %i
			print(im)
			path_read = '/home/azh2/Desktop/CycleGAN-master/prostate_data/adc2/raw/trainB/'+im+'.png'
			
			img = cv2.imread(path_read,1)
			

			path_write3='/home/azh2/Desktop/CycleGAN-master/prostate_data/adc2/raw1'
			cv2.imwrite(path_write3+'1_'+im+'.png',img)

		
		for j in range(1,x22+1):
			
			im = 'rST0000'+str(a)+' ('+str(x12)+')_%d' %j
			print(im)
			path_read = '/home/azh2/Desktop/CycleGAN-master/prostate_data/adc2/raw/trainB/'+im+'.png'
			
			img = cv2.imread(path_read,1)
			

			path_write3='/home/azh2/Desktop/CycleGAN-master/prostate_data/adc2/raw1'
			cv2.imwrite(path_write3+'1_'+im+'.png',img)





#**************************************************************************************************************************************************

#**************************************************************************************************************************************************




def crop_images_training(a,x11,x12,x21,x22):
	if a in numbers:
		for i in range(1,x21+1):


			im = 'rST0000'+'0'+str(a)+' ('+str(x11)+')_%d' %i

			print(im)
			path_read = '/home/azh2/Desktop/CycleGAN-master/prostate_data/adc/with_black/raw/trainA/'+im+'.png'
			img = cv2.imread(path_read,1)
			
			img1= img[64:192,:] #first cropping-> make image 128x256x3
			#img1=img[:,64:192]  #second cropping -> makes image 128x128x3



			path_write1='/home/azh2/Desktop/CycleGAN-master/prostate_data/adc/no_black/others/rc1/trainA/'
			cv2.imwrite(path_write1+im+'.png',img1)

			
			
		for j in range(1,x22+1):
			im = 'rST0000'+'0'+str(a)+' ('+str(x12)+')_%d' %j

			print(im)
			path_read = '/home/azh2/Desktop/CycleGAN-master/prostate_data/adc/with_black/raw/trainA/'+im+'.png'
			img = cv2.imread(path_read,1)
			
			img1= img[64:192,:] #first cropping-> make image 128x256x3
			#img1=img[:,64:192]  #second cropping -> makes image 128x128x3

			path_write1='/home/azh2/Desktop/CycleGAN-master/prostate_data/adc/no_black/others/rc1/trainA/'
			cv2.imwrite(path_write1+im+'.png',img1)


	else:
		for i in range(1,x21+1):


			im = 'rST0000'+str(a)+' ('+str(x11)+')_%d' %i

			print(im)
			path_read = '/home/azh2/Desktop/CycleGAN-master/prostate_data/adc/with_black/raw/trainA/'+im+'.png'
			
			img = cv2.imread(path_read,1)
			
			img1= img[64:192,:] #first cropping-> make image 128x256x3
			#img1=img[:,64:192]  #second cropping -> makes image 128x128x3


			path_write1='/home/azh2/Desktop/CycleGAN-master/prostate_data/adc/no_black/others/rc1/trainA/'
			cv2.imwrite(path_write1+im+'.png',img1)
		
		for j in range(1,x22+1):
			
			im = 'rST0000'+str(a)+' ('+str(x12)+')_%d' %j

			print(im)
			path_read = '/home/azh2/Desktop/CycleGAN-master/prostate_data/adc/with_black/raw/trainA/'+im+'.png'
			img = cv2.imread(path_read,1)
			
			img1= img[64:192,:] #first cropping-> make image 128x256x3
			#img1=img[:,64:192]  #second cropping -> makes image 128x128x3


			path_write1='/home/azh2/Desktop/CycleGAN-master/prostate_data/adc/no_black/others/rc1/trainA/'
			cv2.imwrite(path_write1+im+'.png',img1)


def crop_images_testttt(a,x1,x2):
	for i in range(1,x2+1):


		im = 'rST0000'+str(a)+' ('+str(x1)+')_%d' %i
		print(im)

		
		path_read = '/home/azh2/Desktop/CycleGAN-master/prostate_data/adc/no_black/others/raw/trainA/'+im+'.png'
		img = cv2.imread(path_read,1)


		img1= img[64:192,:] #first cropping-> make image 128x256x3
		#img1=img[:,64:192]  #second cropping -> makes image 128x128x3
		
		path_write1='/home/azh2/Desktop/CycleGAN-master/prostate_data/adc/with_black/raw_cropped1/testB/'
		cv2.imwrite(path_write1+im+'.png',img1)
		






#**************************************************************************************************************************************************
#**************************************************************************************************************************************************

def make_image_binary_testtt(a,x1,x2):

	for i in range(1,x2+1):

		im = 'nST0000'+str(a)+' ('+str(x1)+')_%d' %i
		print(im)

		
		path_read = '/home/azh2/Desktop/CycleGAN-master/prostate_data/adc/noisy+raw/testB/'+im+'.png'
		


		img= cv2.imread(path_read,0)
		ret,img1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
		ret,img2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
			
		for i in range(0,256):
			for j in range(0,256):
				if img1[i][j]== 255:
					img2[i][j]=0
				if img1[i][j]==0:
					img2[i][j]=255

		path_write1='/home/azh2/Desktop/CycleGAN-master/prostate_data/adc/noisy+raw_binary/testB/'
		cv2.imwrite(path_write1+im+'.png',img2)




def make_image_binaryyy(a,x11,x12,x21,x22):
	if a in numbers:
		for i in range(1,x21+1):


			im = 'nST0000'+'0'+str(a)+' ('+str(x11)+')_%d' %i

			print(im)
			path_read = '/home/azh2/Desktop/CycleGAN-master/prostate_data/adc/noisy+raw_cropped/trainB/'+im+'.png'
			img= cv2.imread(path_read,0)
			ret,img1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
			ret,img2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
			
			for i in range(0,128):
				for j in range(0,256):
					if img1[i][j]== 255:
						img2[i][j]=0
					if img1[i][j]==0:
						img2[i][j]=255


			path_write1='/home/azh2/Desktop/CycleGAN-master/prostate_data/adc/noisy+raw_cropped_binary/trainB/'
			cv2.imwrite(path_write1+im+'.png',img2)

			
			
		for j in range(1,x22+1):
			im = 'nST0000'+'0'+str(a)+' ('+str(x12)+')_%d' %j

			print(im)
			path_read = '/home/azh2/Desktop/CycleGAN-master/prostate_data/adc/noisy+raw_cropped/trainB/'+im+'.png'
			img= cv2.imread(path_read,0)
			ret,img1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
			ret,img2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
			
			for i in range(0,128):
				for j in range(0,256):
					if img1[i][j]== 255:
						img2[i][j]=0
					if img1[i][j]==0:
						img2[i][j]=255


			path_write1='/home/azh2/Desktop/CycleGAN-master/prostate_data/adc/noisy+raw_cropped_binary/trainB/'
			cv2.imwrite(path_write1+im+'.png',img2)


	else:
		for i in range(1,x21+1):


			im = 'nST0000'+str(a)+' ('+str(x11)+')_%d' %i

			print(im)

			path_read = '/home/azh2/Desktop/CycleGAN-master/prostate_data/adc/noisy+raw_cropped/trainB/'+im+'.png'
			img= cv2.imread(path_read,0)
			ret,img1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
			ret,img2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
			
			for i in range(0,128):
				for j in range(0,256):
					if img1[i][j]== 255:
						img2[i][j]=0
					if img1[i][j]==0:
						img2[i][j]=255


			path_write1='/home/azh2/Desktop/CycleGAN-master/prostate_data/adc/noisy+raw_cropped_binary/trainB/'
			cv2.imwrite(path_write1+im+'.png',img2)


		
		for j in range(1,x22+1):
			
			im = 'nST0000'+str(a)+' ('+str(x12)+')_%d' %j

			print(im)

			path_read = '/home/azh2/Desktop/CycleGAN-master/prostate_data/adc/noisy+raw_cropped/trainB/'+im+'.png'
			img= cv2.imread(path_read,0)
			ret,img1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
			ret,img2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
			
			for i in range(0,128):
				for j in range(0,256):
					if img1[i][j]== 255:
						img2[i][j]=0
					if img1[i][j]==0:
						img2[i][j]=255


			path_write1='/home/azh2/Desktop/CycleGAN-master/prostate_data/adc/noisy+raw_cropped_binary/trainB/'
			cv2.imwrite(path_write1+im+'.png',img2)









numbers=[0,1,2,3,4,5,6,7,8,9]

'''
#ADC

for a in xrange(0,16):
	if a==0:
		x11=1
		x21=16

		x12=34
		x22=16
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==1:
		x11=10
		x21=16

		x12=35
		x22=22
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==2:
		x11=11
		x21=16

		x12=36
		x22=32
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==3:
		x11=12
		x21=16

		x12=37
		x22=32
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==4:
		x11=13
		x21=16

		x12=38
		x22=30
		separate_images_traininggg(a,x11,x12,x21,x22)

	if a==5:
		x11=14
		x21=16

		x12=39
		x22=32
		separate_images_traininggg(a,x11,x12,x21,x22)

	if a==6:
		x11=4
		x21=16

		x12=15
		x22=16
		separate_images_traininggg(a,x11,x12,x21,x22)

	if a==7:
		x11=16
		x21=25

		x12=40
		x22=16
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==8:
		x11=5
		x21=16

		x12=17
		x22=32
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==9:
		x11=6
		x21=16

		x12=18
		x22=16
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==10:
		x11=7
		x21=32

		x12=19
		x22=16
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==11:
		x11=2
		x21=16

		x12=8
		x22=32
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==12:
		x11=9
		x21=32

		x12=20
		x22=32
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==13:
		x11=21
		x21=32

		x12=28
		x22=32
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==14:
		x11=22
		x21=30

		x12=0
		x22=0
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==15:
		x11=23
		x21=16

		x12=0
		x22=0
		separate_images_traininggg(a,x11,x12,x21,x22)

'''

'''
#ADC and DWI
for a in range(16,26):
	if a==16:
		x1=24
		x2=16
		separate_images_test(a,x1,x2)
	
	if a==17:
		x1=25
		x2=16
		separate_images_test(a,x1,x2)	
	
	if a==18:
		x1=26
		x2=16
		separate_images_test(a,x1,x2)
	
	if a==19:
		x1=27
		x2=32
		separate_images_test(a,x1,x2)
	
	if a==20:
		x1=29
		x2=32
		separate_images_test(a,x1,x2)
	
	if a==21:
		x1=3
		x2=16
		separate_images_test(a,x1,x2)
	
	if a==22:
		x1=30
		x2=32
		separate_images_test(a,x1,x2)
	
	if a==23:
		x1=31
		x2=32
		separate_images_test(a,x1,x2)
	
	if a==24:
		x1=32
		x2=32
		separate_images_test(a,x1,x2)
	
	if a==25:
		x1=33
		x2=16
		separate_images_test(a,x1,x2)


'''


'''

#T2W :

for a in xrange(0,16):
	if a==0:
		x11=1
		x21=30

		x12=34
		x22=19
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==1:
		x11=10
		x21=26

		x12=35
		x22=30
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==2:
		x11=11
		x21=30

		x12=36
		x22=26
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==3:
		x11=12
		x21=25

		x12=37
		x22=30
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==4:
		x11=13
		x21=30

		x12=38
		x22=30
		separate_images_traininggg(a,x11,x12,x21,x22)

	if a==5:
		x11=14
		x21=30

		x12=39
		x22=30
		separate_images_traininggg(a,x11,x12,x21,x22)

	if a==6:
		x11=4
		x21=30

		x12=15
		x22=30
		separate_images_traininggg(a,x11,x12,x21,x22)

	if a==7:
		x11=16
		x21=25

		x12=40
		x22=24
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==8:
		x11=5
		x21=27

		x12=17
		x22=30
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==9:
		x11=6
		x21=30

		x12=18
		x22=20
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==10:
		x11=7
		x21=30

		x12=19
		x22=30
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==11:
		x11=2
		x21=30

		x12=8
		x22=30
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==12:
		x11=9
		x21=30

		x12=20
		x22=30
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==13:
		x11=21
		x21=32

		x12=28
		x22=37
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==14:
		x11=22
		x21=30

		x12=0
		x22=0
		separate_images_traininggg(a,x11,x12,x21,x22)
	if a==15:
		x11=23
		x21=26

		x12=0
		x22=0
		separate_images_traininggg(a,x11,x12,x21,x22)


'''


#T2W

for a in range(16,26):
	if a==16:
		x1=24
		x2=30
		separate_images_test(a,x1,x2)
	
	if a==17:
		x1=25
		x2=30
		separate_images_test(a,x1,x2)	
	
	if a==18:
		x1=26
		x2=30
		separate_images_test(a,x1,x2)
	
	if a==19:
		x1=27
		x2=30
		separate_images_test(a,x1,x2)
	
	if a==20:
		x1=29
		x2=30
		separate_images_test(a,x1,x2)
	
	if a==21:
		x1=3
		x2=28
		separate_images_test(a,x1,x2)
	
	if a==22:
		x1=30
		x2=34
		separate_images_test(a,x1,x2)
	
	if a==23:
		x1=31
		x2=32
		separate_images_test(a,x1,x2)
	
	if a==24:
		x1=32
		x2=32
		separate_images_test(a,x1,x2)
	
	if a==25:
		x1=33
		x2=22
		separate_images_test(a,x1,x2)