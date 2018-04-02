import pandas as pd
import numpy as np
import xlrd
import matplotlib.pyplot as plt

# read the excel file
b = pd.read_excel('~/Desktop/kc_house_data.xlsx')

# store the data as numpy ndarray
b = b.values
price = b[:,4]
b = b[:,[0, 1, 2, 3]]

# Adding a column containing ones for the sake of theta[0]
x = np.ones((b.shape[0],b.shape[1]+31))
x[:,:-31] = b
x[:,5] = np.multiply(x[:,0], x[:,0])	#a*a
x[:,6] = np.multiply(x[:,1], x[:,1])	#b*b
x[:,7] = np.multiply(x[:,2], x[:,2])	#c*c
x[:,8] = np.multiply(x[:,3], x[:,3])	#d*d
x[:,9] = np.multiply(x[:,0], x[:,1])	#a*b
x[:,10] = np.multiply(x[:,0], x[:,2])	#a*c
x[:,11] = np.multiply(x[:,0], x[:,3])	#a*d
x[:,12] = np.multiply(x[:,1], x[:,2])	#b*c
x[:,13] = np.multiply(x[:,1], x[:,3])	#b*d
x[:,14] = np.multiply(x[:,2], x[:,3])	#c*d
x[:,15] = np.multiply(x[:,5], x[:,0])	#a*a*a
x[:,16] = np.multiply(x[:,6], x[:,1])	#b*b*b
x[:,17] = np.multiply(x[:,7], x[:,2])	#c*c*c
x[:,18] = np.multiply(x[:,8], x[:,3])	#d*d*d
x[:,19] = np.multiply(x[:,5], x[:,1])	#a*a*b
x[:,20] = np.multiply(x[:,5], x[:,2])	#a*a*c
x[:,21] = np.multiply(x[:,5], x[:,3])	#a*a*d
x[:,22] = np.multiply(x[:,6], x[:,0])	#b*b*a
x[:,23] = np.multiply(x[:,6], x[:,2])	#b*b*c
x[:,24] = np.multiply(x[:,6], x[:,3])	#b*b*d
x[:,25] = np.multiply(x[:,7], x[:,0])	#c*c*a
x[:,26] = np.multiply(x[:,7], x[:,1])	#c*c*b
x[:,27] = np.multiply(x[:,7], x[:,3])	#c*c*d
x[:,28] = np.multiply(x[:,8], x[:,0])	#d*d*a
x[:,29] = np.multiply(x[:,8], x[:,1])	#d*d*b
x[:,30] = np.multiply(x[:,8], x[:,2])	#d*d*c
x[:,31] = np.multiply(x[:,9], x[:,2])	#a*b*c
x[:,32] = np.multiply(x[:,9], x[:,2])	#a*b*d
x[:,33] = np.multiply(x[:,10], x[:,3])	#a*c*d
x[:,34] = np.multiply(x[:,12], x[:,3])	#b*c*d

# print(type(b))
# <class 'numpy.ndarray'>

# print(b)

# mean normalization of data
for i in range(35):
	if i!=4:
		x[:, i] = np.subtract(x[:, i], np.sum(x, axis=0)[i]/x.shape[0])

# feature scaling the attributes
for i in range(35):
	if i != 4:
		x[:, i] = np.divide(x[:, i], np.std(x[:, i]))

# test = last 20% of the given data
# x = first 80% of the given data
n_x = (int)(0.8*x.shape[0])
test = x[n_x:, :]
x = x[:n_x, :]

# hypothesis function:
# = theta[0] * x[i][0] + theta[1] * x[i][1] + theta[2] * x[i][2] +
#   theta[3] * x[i][3] + theta[4] * x[i][4]

# function for evaluation of summation of (h-theta(x[i]) - y[i]) * x[i][0-4]
def hypothesis(theta, x, y):
	summation = 0
	for i in range(x.shape[0]):
		summation += (np.dot(theta, x[i]) - y[i]) * x[i]
	return summation

# use gradient descent to minimize cost and obtain theta values

# List of different Values of alpha 
alpha_list = [0.01, 0.02, 0.05, 0.1, 0.15, 0.20]
err = []

for alpha in alpha_list:
	theta = np.random.rand(35)
	# repeat this until convergence
	cnt = 0
	while cnt < 50:
		temp_theta = np.subtract(theta, alpha * hypothesis(theta, x, price) / x.shape[0])
		if np.sum(abs(theta - temp_theta)) < 0.01:
			theta = temp_theta
			break;
		theta = temp_theta
		cnt += 1

	# test the accuracy:
	summation = 0
	#print(theta)
	for i in range(test.shape[0]):
		summation += (np.dot(theta, test[i]) - price[i+n_x]) ** 2
	summation = summation / (2 * test.shape[0])
	RMSE = summation ** 0.5
	print("Finally Learned Values of theta are:")
	print(theta)
	print("RMSE finally obtained is %s" % (RMSE))
	err.append(RMSE)

# Plot using matplotlib RMSE vs lambda
plt.plot(alpha_list, err, '-o')
plt.axis([0, 0.21, 0, 1000000])
plt.xlabel('Learning Rate')
plt.ylabel('RMSE')
plt.show()

