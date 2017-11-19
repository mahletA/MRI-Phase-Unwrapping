import numpy as np

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
					unwrappedX[i][j] = image[i,j] + xGrad[i][j] + 2*pi

				elif (xGrad[i][j] >= pi):
					unwrappedX[i][j] = image[i,j] + xGrad[i][j] - 2*pi

				elif (xGrad[i][j] > -pi and xGrad[i][j] < pi):
					unwrappedX[i][j] = image[i,j] + xGrad[i][j]

				else :
					print "There is some weird error in Y gradient"

			
	for i in range(0,M):
		for j in range(0,N):

			#copy the last column of the image as it is to the unwrapped phase estimate
			if (j == N):
				unwrappedY[i][j] = image[i,j]

			else :
				if (yGrad[i][j] <= -pi):
					unwrappedY[i][j] = image[i,j] + yGrad[i][j] + 2*pi

				elif (yGrad[i][j] >= pi):
					unwrappedY[i][j] = image[i,j] + yGrad[i][j] - 2*pi

				elif (yGrad[i][j] > -pi and yGrad[i][j] < pi):
					unwrappedY[i][j] = image[i,j] + yGrad[i][j]

				else :
					print "There is some weird error in X gradient"

	return unwrappedX , unwrappedY


def zValCalc (xGrad , yGrad , window ,M , N):
	
	if ( M == N):
		zValues = [0] * ( (M * N) / (window * window) )
		xMean = [0] * ( (M * N) / (window * window) )
		yMean = [0] * ( (M * N )/ (window * window) )

	xGrad_copy = xGrad
	yGrad_copy = yGrad

	xGrad_copy = xGrad_copy.append(np.zeros(M))

	for i in range(0,len(yGrad_copy)):
		i.append(0)

	counter = -1
	jCounter = -1
	flag = False

	for i in range(0,M):

		if ( i % window == 0):
			flag = True;

		for j in range(0,N):

			if (i%window == 0 and j%window == 0):
				counter += 1
				

			if (flag == True):
				xMean[counter] += xGrad_copy[i][j]
				yMean[counter] += yGrad_copy[i][j]

			else :

				if (j%window == 0):
					jCounter += 1
				xMean[jCounter] += xGrad_copy[i][j]
				yMean[jCounter] += yGrad_copy[i][j]

		flag = False

	xMean = [float(x) / (window * window) for x in xMean]
	yMean = [float(y) / (window * window) for y in yMean]


	return xMean ,yMean



	#def Weights (zValues ):

	#a = [[1,1,1,1],[1,1,1,0],[1,1,1,0],[0,0,0,0]]
	#b = [[2,2,2,2,2,2,2,2],[2,2,2,2,2,2,2,2],[2,2,2,2,2,2,2,2],[2,2,2,2,2,2,2,2],[2,2,2,2,2,2,2,2],[2,2,2,2,2,2,2,2],[2,2,2,2,2,2,2,2],[2,2,2,2,2,2,2,2]]
	#aMean,bMean = zValCalc(a , a ,2 , 4, 4)
	#print aMean

