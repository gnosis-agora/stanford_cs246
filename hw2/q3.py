import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def init_P_Q(file_dir,k):
	"""
	initialize matrices P and Q with random weights between (0,(5/k)**0.5)
	n = number of users
	m = number of movies
	k = number of factors
	"""
	Q = {}
	P = {}
	multiplier = np.sqrt((5.0/float(k)))
	with open(file_dir) as f:
		for line in f.readlines():
			user,movie,rating = [int(val) for val in line.split()]
			if user in P:
				pass
			else:
				P[user] = np.random.rand(k) * multiplier
			if movie in Q:
				pass
			else:
				Q[movie] = np.random.rand(k) * multiplier
	return P,Q

def get_error(rating, q_i, p_u):
	"""
	Returns the error score for (i,u) entry in matrix R
	rating = true movie rating score that user u gave for movie i
	q_i = ith row of matrix Q
	p_u = uth row of matrix P
	"""
	return (rating - np.dot(q_i, np.transpose(p_u)))**2

def run_iter(file_dir, P, Q, l, u):
	"""
	Performs 1 iteration of stochastic gradient descent algorithm
	"""
	with open(file_dir) as f:
		lines = f.readlines()
		for idx in range(len(lines)):
			user,movie,rating = [int(val) for val in lines[idx].split()]
			# Do SGD
			error_deriv = 2.0 * (rating - np.dot(Q[movie], np.transpose(P[user])))
			new_Q = Q[movie] + u * (error_deriv * P[user] - 2.0 * l * Q[movie])
			new_P = P[user] + u * (error_deriv * Q[movie] - 2.0 * l * P[user])
			Q[movie] = new_Q
			P[user] = new_P
		# Calculate error after full iteration
		error = 0.0
		for idx in range(len(lines)):
			user,movie,rating = [int(val) for val in lines[idx].split()]
			# Add to total error
			# print rating,user,movie,np.dot(Q[movie], np.transpose(P[user]))
			error += get_error(rating, Q[movie], P[user])
		for user in P:
			error += l * np.linalg.norm(P[user])**2
		for user in Q:
			error += l * np.linalg.norm(Q[movie])**2
		return P,Q,error


file_dir = './q3/data/ratings.train.txt'
u = 0.015
k = 20
l = 0.1
P,Q = init_P_Q(file_dir, k)
error_list = []
for i in range(40):
	P,Q,error = run_iter(file_dir, P, Q, l, u)
	error_list.append(error)
	print "Error after {0} iteration: {1}".format(i+1, error)

x = [i+1 for i in range(40)]
y = error_list
plt.plot(x, y, "-o")
plt.xlabel("# of Iteration")
plt.ylabel("Error")
plt.title("Error vs Iteration")
plt.show()
