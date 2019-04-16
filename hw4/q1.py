import numpy as np
import copy
import time
import matplotlib.pyplot as plt

def get_data(feature_dir, target_dir):
	"""
	Returns matrix representations of 
	feature shape = 6414 * 122
	target shape = 6414 * 1
	"""
	features = []
	with open(feature_dir) as f:
		for line in f.readlines():
			features.append([float(ele) for ele in line.strip().split(',')])
	targets = []
	with open(target_dir) as f:
		for line in f.readlines():
			targets.append(float(line.strip()))
	return np.matrix(features), np.transpose(np.matrix(targets))

# Batch gradient descent
def update_w(C, w, b, features, targets, l):
	"""
	Computes the gradient of every element in the weight matrix and updates them
	C = regularisation factor
	w = weight matrix
	b = constant factor
	features, targets = matrix of attributes and label
	l = learning rate
	"""
	slack_sum = np.zeros(features.shape[1])
	for i in range(len(features)):
	    x = features[i]
	    y = float(targets[i])
	    xw = sum(np.dot(x,w))
	    cond = y * (xw + b)
	    if cond >= 1:
	        pass
	    else:
	        slack_sum -= np.multiply(y,np.squeeze(np.asarray(x)))

	new_w = w - l * (w + C * slack_sum)
	return new_w

def update_b(w, b, C, l, features, targets):
	"""
	Same as function above but updates b instead of w
	"""
	evaluation_matrix = np.multiply(targets,np.transpose(np.dot(features,w)) + b)
	evaluation_matrix = np.where(evaluation_matrix < 1, evaluation_matrix, 0)
	evaluation_matrix = np.where(evaluation_matrix == 0, evaluation_matrix, -targets)
	new_b = b - l * C * float(sum(evaluation_matrix))
	return new_b

def get_cost(w, b, C, features, targets):
	"""
	Computes the cost in terms of error of current weights
	C = regularisation factor
	w = weight matrix
	b = constant factor
	features, targets = matrix of attributes and label
	"""
	weight_sum = 0.5 * float(np.dot(w,np.transpose(w)))
	evaluation_matrix = np.multiply(targets,np.transpose(np.dot(features,w)) + b)
	evaluation_matrix = np.where(evaluation_matrix < 1, evaluation_matrix, 0)
	evaluation_matrix = np.where(evaluation_matrix == 0, evaluation_matrix, 1 - evaluation_matrix)
	slack_sum = C * float(sum(evaluation_matrix))	
	return weight_sum + slack_sum

def get_cost_change(old_cost, new_cost):
	"""
	Returns the cost change in %
	"""
	return (abs(new_cost - old_cost) * 100) / old_cost

start = time.time()
l = 0.0000003
e = 0.25
b = 0
features,targets = get_data('./q1/data/features.txt','./q1/data/target.txt')
w = np.zeros(len(np.transpose(features[0])))
C = 100
k = 0
BGD_cost_history = []
new_cost = get_cost(w, b, C, features, targets)

while 1:
	w = update_w(C, w, b, features, targets, l)
	b = update_b(w, b, C, l, features, targets)
	new_cost = get_cost(w, b, C, features, targets)
	BGD_cost_history.append(new_cost)
	if k == 0:
		pass
	else:
		cost_change = get_cost_change(BGD_cost_history[k-1], BGD_cost_history[k])
		if cost_change < e:
			break
	k += 1
	# if k % 10 == 0:
	# 	print "iter {} cost: {} change %: {:0.10f}".format(k,new_cost,cost_change)
print "Batch gradient descent"
# print BGD_cost_history
elapsed = time.time() - start
print elapsed

# Stochastic Gradient Descent
def update_w_sgd(C, w, b, i, features, targets, l):
	"""
	Computes the gradient of every element in the weight matrix and updates them
	C = regularisation factor
	w = weight matrix
	b = constant factor
	i = chosen index of comparison
	features, targets = matrix of attributes and label
	l = learning rate
	"""
	slack_sum = np.zeros(features.shape[1])
	x = features[i]
	y = float(targets[i])
	xw = sum(np.dot(x,w))
	cond = y * (xw + b)
	if cond >= 1:
	    pass
	else:
	    slack_sum -= np.multiply(y,np.squeeze(np.asarray(x)))
	new_w = w - l * (w + C * slack_sum)
	return new_w

def update_b_sgd(w, b, C, l, i, features, targets):
	"""
	Same as function above but updates b instead of w
	"""
	x = features[i]
	y = float(targets[i])
	xw = sum(np.dot(x,w))
	cond = y * (xw + b)
	if cond >= 1:
	    slack_sum = 0
	else:
	    slack_sum = -y
	new_b = b - l * C * slack_sum
	return new_b

def get_cost_change_sgd(old_cost, new_cost, old_cost_change):
	"""
	Returns the cost change 
	"""
	if old_cost_change == 0:
		return ((abs(new_cost - old_cost) * 100) / old_cost)
	return 0.5 * old_cost_change + 0.5 * ((abs(new_cost - old_cost) * 100) / old_cost)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

start = time.time()
l = 0.0001
e = 0.001
b = 0
features,targets = get_data('./q1/data/features.txt','./q1/data/target.txt')
features,targets = unison_shuffled_copies(features, targets)
w = np.zeros(features.shape[1])
C = 100
k = 0
i = 0
SGD_cost_history = []
new_cost = get_cost(w, b, C, features, targets)
cost_change = 0

while 1:
	w = update_w_sgd(C, w, b, i, features, targets, l)
	b = update_b_sgd(w, b, C, l, i, features, targets)
	new_cost = get_cost(w, b, C, features, targets)
	SGD_cost_history.append(new_cost)

	if k == 0:
		pass
	else:
		cost_change = get_cost_change_sgd(SGD_cost_history[k-1], SGD_cost_history[k], cost_change)
		if cost_change < e:
			break
	i = (i % features.shape[0]) + 1 
	k += 1
	# if k % 10 == 0:
	# 	print "iter {} cost: {} change %: {:0.10f}".format(k,new_cost,cost_change)
print "Stochastic gradient descent"
# print SGD_cost_history
elapsed = time.time() - start
print elapsed

# Mini batch gradient descent
start = time.time()
l = 0.00001
e = 0.01
b = 0
features,targets = get_data('./q1/data/features.txt','./q1/data/target.txt')
features,targets = unison_shuffled_copies(features, targets)
w = np.zeros(features.shape[1])
C = 100
k = 0
MBGD_cost_history = []
new_cost = get_cost(w, b, C, features, targets)
cost_change = 0
L = 0
batch_size = 20
n = features.shape[0]

while 1:
	batch_features = features[(L * batch_size): min(n,(L+1) * batch_size) ,]
	batch_targets = targets[(L * batch_size): min(n,(L+1) * batch_size) ,]
	w = update_w(C, w, b, batch_features, batch_targets, l)
	b = update_b(w, b, C, l, batch_features, batch_targets)
	new_cost = get_cost(w, b, C, features, targets)
	MBGD_cost_history.append(new_cost)

	if k == 0:
		pass
	else:
		cost_change = get_cost_change_sgd(MBGD_cost_history[k-1], MBGD_cost_history[k], cost_change)
		if cost_change < e:
			break
	L = (L+1) % ((n+batch_size - 1)/batch_size)
	k += 1
	# if k % 10 == 0:
	# 	print "iter {} cost: {} change %: {:0.10f}".format(k,new_cost,cost_change)
print "Mini batch gradient descent"
# print MBGD_cost_history
elapsed = time.time() - start
print elapsed

plt.plot(BGD_cost_history, label='BGD')
plt.plot(SGD_cost_history, label='SGD')
plt.plot(MBGD_cost_history, label='mini BGD')
plt.xlabel('iterations')
plt.ylabel('cost')
plt.legend()
plt.show()








