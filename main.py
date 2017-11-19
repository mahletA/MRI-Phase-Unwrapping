# Matplotlib library
import matplotlib.pyplot as plt
from matplotlib import cm

# Numpy library
import numpy as np

from skimage import io
from scipy import signal

# Functions for various calculations

def Gradient(image):
	M , N = np.shape(image)

	xGrad = np.zeros((M-1, N) , dtype = float)
	yGrad = np.zeros((M, N-1) , dtype = float)

	for i in range(0,M-1):
		for j in range(0,N):
			xGrad[i][j] = image[i+1][j] - image[i][j]

	for i in range(0,M):
		for j in range(0,N-1):
			yGrad[i][j] = image[i][j+1] - image[i][j]

	return xGrad , yGrad

def UnwrapedEstimate (image , xGrad ,yGrad):
	pi = 0.5*(np.max(image))

	unwrappedX= np.zeros((M, N) , dtype = float)
	unwrappedY = np.zeros((M, N) , dtype = float)

	for i in range(0,M):
		for j in range(0,N):

			#copy the last row of the image as it is to the unwrapped phase estimate
			if (i == M-1):
				unwrappedX[i][j] = image[i,j]

			else :
				if (xGrad[i][j] <= -pi):
					print "case 1 ----------X"
					unwrappedX[i][j] = image[i,j] + xGrad[i][j] + 2*pi

				elif (xGrad[i][j] >= pi):
					#print "case 2 ----------X"
					unwrappedX[i][j] = image[i,j] + xGrad[i][j] - 2*pi

				elif (xGrad[i][j] > -pi and xGrad[i][j] < pi):
					#print "case 3 ----------X"
					unwrappedX[i][j] = image[i,j] + xGrad[i][j]

				else :
					print "There is some weird error in the X Gradient"


	for i in range(0,M):
		for j in range(0,N):

			#copy the last row of the image as it is to the unwrapped phase estimate
			if (j == N-1):
				unwrappedX[i][j] = image[i,j]

			else :
				if (yGrad[i][j] <= -pi):
					print "case 1 ----------Y"
					unwrappedY[i][j] = image[i,j] + yGrad[i][j] + 2*pi

				elif (yGrad[i][j] >= pi):
					#print "case 2 ----------Y"
					unwrappedY[i][j] = image[i,j] + yGrad[i][j] - 2*pi

				elif (yGrad[i][j] > -pi and yGrad[i][j] < pi):
					#print "case 3 ----------Y"
					unwrappedY[i][j] = image[i,j] + yGrad[i][j]

				else :
					print "There is some weird error in the Y Gradient"

		
	

	return unwrappedX ,unwrappedY




if __name__ == "__main__":

	image_path = './Images/'
	image_name='607_0001.bmp'

	
	wrapped=io.imread(image_path.__add__(image_name))

	M , N = np.shape(wrapped)
	xGrad , yGrad = Gradient(wrapped)

	estX ,estY = UnwrapedEstimate (wrapped ,xGrad ,yGrad)
	xGradEst , yGradEst = Gradient(estY)

	fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 10))   
	ax0, ax1 ,ax2 ,ax3 ,ax4 ,ax5 ,ax6 ,ax7 = axes.ravel()

	print np.shape(estX)
	print np.shape (estY)

	ax0.imshow(wrapped) 
	ax1.imshow(wrapped)
	ax2.imshow(xGrad) 
	ax3.imshow(yGrad) 
	ax4.imshow(estX)
	ax5.imshow(estY) 
	ax6.imshow(xGradEst) 
	ax7.imshow(yGradEst)
	plt.show()

	#Hx = np.multiply([-1 , 0 , 1],0.5)
	#Hy = np.transpose(Hx)

	#xGrad_python = signal.convolve2d(wrapped , Hx)
	#yGrad_python = signal.convolve2d(wrapped , Hy)

	#io.imshow(xGrad)
	#plt.show()

	#io.imshow(yGrad)
	#plt.show()