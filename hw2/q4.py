import numpy as np
from scipy import linalg
# import matplotlib.pyplot as plt

shows_dir = './q4/data/shows.txt'
user_shows_dir = './q4/data/user-shows.txt'


def init_P_Q(R):
	row_sum = []
	col_sum = []
	for row in R:
		row_sum.append(np.sum(row))
	for col in np.transpose(R):
		col_sum.append(np.sum(col))
	return np.diag(row_sum), np.diag(col_sum)

def factorize(x):
	return np.where(x == 0, 0, x**-0.5)

def i_i_gamma(R, Q):
	sqrt_Q = factorize(Q)
	return np.linalg.multi_dot([R, sqrt_Q, np.transpose(R), R, sqrt_Q])

def u_u_gamma(R, P):
	sqrt_P = factorize(P)
	return np.transpose(np.linalg.multi_dot([np.transpose(R), sqrt_P, R, np.transpose(R), sqrt_P]))

def get_top_indexes(arr, n):
	"""
	Returns the index of the top n elements in an array
	"""
	processed_arr = [(arr[i], i) for i in range(len(arr))]
	processed_arr = sorted(processed_arr, key=lambda ele: (-ele[0],ele[1]))
	return [ele[1] for ele in processed_arr][:n]

def get_show_titles(shows_dir):
	shows = []
	with open(shows_dir) as f:
		for line in f.readlines():
			shows.append(line.strip("\"\n"))
	return shows
R = np.loadtxt(user_shows_dir)
P,Q = init_P_Q(R)
u_gamma = u_u_gamma(R, P)
i_gamma = i_i_gamma(R, Q)
u_top_indexes = get_top_indexes(u_gamma[499][:100], 5)
i_top_indexes = get_top_indexes(i_gamma[499][:100], 5)
show_titles = get_show_titles(shows_dir)

print u_top_indexes
print i_top_indexes
print "top 5 shows by user-user collaborative filtering: {0}".format(list(map(lambda ele: show_titles[ele], u_top_indexes)))
print "top 5 shows by item-item collaborative filtering: {0}".format(list(map(lambda ele: show_titles[ele], i_top_indexes)))