import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os 


# The accuracy is 95% with 250 iteration using neural network 
#  with backpropogation "https://www.kaggle.com/primaryobjects/voicegender"
# which is 98% on training dataset which is above baseline.

path = os.getcwd()
path = path + '/voice.csv'
data = pd.read_csv(path,header = None)
#print(data)
print(data.head())
print(data.shape)
#data.drop(data.index[[0]],inplace = True)
cols = data.shape[1]
X = data.iloc[1:len(data),0:cols-1]
X = np.matrix(X)
X = X.astype(np.float)
#print(X)

#rows = data.shape[0]
#print(cols)
#s = data.iloc[0,0]
#print(s)
#X = data.iloc[:,0:cols-1]    # array size or list size 3168 rows and 20 columns 
#print(X)
#X = float(X)
y = data.iloc[:,cols-1:cols]  # array size or list size 3168 rows and 1 columns
#print(y)
#print(X.shape[1])  # it prints number of columns 

y = np.array(y)
#print(y)

# 1 for male ; 2 for female 
for i in range(len(y)):
	if y[i][0] == 'male':
		y[i][0] = 1
	else:
		y_onehot[i][0] = 2  # female

#print(y)=
#print(y.shape)

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse = False)
y_onehot = encoder.fit_transform(y)
print(y_onehot.shape) # 3168*2
print(y[0],y_onehot[3167,:])
X = np.array(X)
print(X.shape)	# 3168*20


def sigmoid(z):
	return 1/(1+np.exp(-z))

def forward_propogate(X,theta1,theta2):

	m = X.shape[0]  # Number of Rows of actual data

	a1 = np.insert(X,0,values = np.ones(m),axis = 1)
	# 3168*21 ---> a1 dimension , 5*21 ---> theta1 dimension
	z2 = a1*theta1.T      # z2 ---> 3168 * 5 
	#a2 ---> 3168 * 6 , theta2 = 2 * 6
	a2 = np.insert(sigmoid(z2),0,values = np.ones(m),axis = 1) 
	z3 = a2 * theta2.T  # z3 dimension ----> 3168 * 2
	# dimension of h ---> 3168 * 2 which is similar to y_onehot
	h = sigmoid(z3)


	return a1,z2,a2,z3,h 


def cost(params,input_size,hidden_size,num_labels,X,y,learning_rate):

	m = X.shape[0]     #Number of rows --> m = 3168
	X = np.matrix(X)  #already done in this case 
	y = np.matrix(y)  #already done in this case 

	theta1 = np.matrix(np.reshape(params[:hidden_size*(input_size+1)],(hidden_size,(input_size+1))))
	theta2 = np.matrix(np.reshape(params[hidden_size*(input_size+1):],(num_labels,(hidden_size+1))))

	a1,z2,a2,z3,h = forward_propogate(X,theta1,theta2)

	J = 0
	for  i in range(m):
		first_term = np.multiply(-y[i,:],np.log(h[i,:]))
		second_term = np.multiply((1-y[i,:]),np.log(1-h[i,:]))
		J += np.sum(first_term - second_term)
		J = J/m
		J += (float(learning_rate)/(2*m))*(np.sum(np.power(theta1[:,1:],2)) + np.sum(np.power(theta2[:,1:],2)))

		return J

input_size = 20
hidden_size = 5
num_labels = 2
learning_rate = 0.55

params = (np.random.random(size = hidden_size*(input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25
m = X.shape[0]
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)],(hidden_size,(input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):],(num_labels,(hidden_size + 1))))

print(theta1.shape,theta2.shape)
 

a1,z2,a2,z3,h = forward_propogate(X,theta1,theta2)
print(a1.shape,z2.shape,a2.shape,z3.shape,h.shape)

#print(theta1)
#print(X)
#a1 = np.insert(X,0,values = np.ones(m),axis = 1)
#print(a1)
#print(theta1)
#print(y)
#print(type(X))
#print(type(y))
#print(X.dtype)

print(cost(params,input_size,hidden_size,num_labels,X,y_onehot,learning_rate))

def sigmoid_gradient(z):
	return np.multiply(sigmoid(z),(1-sigmoid(z)))


def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):  
    ##### this section is identical to the cost function logic we already saw #####
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propogate(X, theta1, theta2)

    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)

    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)

    J = J / m

    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))

    ##### end of cost function logic, below is the new part #####

    # perform backpropagation
    for t in range(m):
        a1t = a1[t,:]  # (1, 401)
        z2t = z2[t,:]  # (1, 25)
        a2t = a2[t,:]  # (1, 26)
        ht = h[t,:]  # (1, 10)
        yt = y[t,:]  # (1, 10)

        d3t = ht - yt  # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)

        delta1 = delta1 + (d2t[:,1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    # add the gradient regularization term
    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * learning_rate) / m
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * learning_rate) / m

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad


J, grad = backprop(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)  

print(J,grad.shape)


from scipy.optimize import minimize

# minimize the objective function
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),  
                method='TNC', jac=True, options={'maxiter': 250})


print(fmin)

X = np.matrix(X)  
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))  
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propogate(X, theta1, theta2)  
y_pred = np.array(np.argmax(h, axis=1) + 1)  

print(y_pred)

correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]  
accuracy = (sum(map(int, correct)) / float(len(correct)))  
print ('accuracy = {0}%'.format(accuracy * 100))






