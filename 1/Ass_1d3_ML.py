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
# print(type(b))
# <class 'numpy.ndarray'>

# print(b)

# mean normalization of data
for i in range(4):
	b[:, i] = np.subtract(b[:, i], np.sum(b, axis=0)[i]/b.shape[0])

# feature scaling the attributes
for i in range(4):
	#b[:, i] = np.divide(b[:, i], np.std(b[:, i]))
	b[:, i] = np.divide(b[:, i], np.amax(b[:, i]) - np.amin(b[:, i]))

print(b)

# Adding a column containing ones for the sake of theta[0]
x = np.ones((b.shape[0],b.shape[1]+1))
x[:,:-1] = b

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
		summation += (np.dot(theta, x[i]) - y[i]) / x.shape[0] * (np.dot(theta, x[i]) - y[i]) * x[i]
	return summation

# List of different Values of alpha 
alpha_list = [0.01, 0.02, 0.05, 0.1]
err = []

# use gradient descent to minimize cost and obtain theta values

for alpha in alpha_list:
	theta = np.random.rand(5)
# repeat this until convergence
	cnt = 0
	while cnt < 5:
		temp_theta = np.subtract(theta, alpha * hypothesis(theta, x, price))
		if np.sum(abs(theta - temp_theta)) < 0.01:
			theta = temp_theta
			break;
		theta = temp_theta
		print(theta)
		cnt += 1

	# test the accuracy:
	summation = 0
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
plt.xlabel('Learning Rate')
plt.ylabel('RMSE')
plt.show()
