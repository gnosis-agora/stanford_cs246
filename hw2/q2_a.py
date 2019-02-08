import sys
from pyspark import SparkConf, SparkContext
import numpy as np
import matplotlib.pyplot as plt

conf = SparkConf()
sc = SparkContext(conf=conf)

lines = sc.textFile("./q2/data/data.txt")
centroid_file = './q2/data/c1.txt'

def eucli_distance(mat1, mat2):
	return np.linalg.norm(mat1-mat2)

def manhattan_distance(mat1, mat2):
	return abs(mat1-mat2).sum()

def get_nearest_centroid(point, centroids_map):
	# euclidean distance
	nearest_centroid_map = min([(key, point, eucli_distance(point,centroids_map[key])**2) for key in centroids_map], key=lambda ele: ele[2])
	# manhattan_distance
	# nearest_centroid_map = min([(key, point, manhattan_distance(point,centroids_map[key])) for key in centroids_map], key=lambda ele: ele[2])
	return nearest_centroid_map

def iterative_k_means(max_iter,centroid_file):
	if max_iter > 20:
		raise IndexError("Max iterations out of acceptable limit")
	# Select k points as centroids
	centroids = np.loadtxt(centroid_file)
	# Use integer for easy centroid representation
	centroids_map = {}
	for idx in range(len(centroids)):
		centroids_map[idx] = centroids[idx] 
	total_cost = 0
	# Start iteration process
	for i in range(max_iter):
		processed_centroid_list = lines.map(lambda l: np.matrix(str(l))) \
		.map(lambda point: get_nearest_centroid(point,centroids_map))

		total_cost = processed_centroid_list.map(lambda ele: ele[2]).sum()

		centroids_count = processed_centroid_list.map(lambda ele: (ele[0],1)) \
		.reduceByKey(lambda c1,c2: c1 + c2).collectAsMap()

		recomputed_centroids = processed_centroid_list.map(lambda ele: (ele[0],ele[1])) \
		.reduceByKey(lambda c1,c2: c1 + c2) \
		.map(lambda ele: (ele[0],ele[1]/centroids_count[ele[0]])).collectAsMap()
		centroids_map = recomputed_centroids
	return total_cost
c1_costs = []
c2_costs = []
for i in range(20):
	c1_costs.append(iterative_k_means(i+1,'./q2/data/c1.txt'))
	c2_costs.append(iterative_k_means(i+1,'./q2/data/c2.txt'))
line1, = plt.plot(c1_costs)
line2, = plt.plot(c2_costs)
print "c1 costs:",c1_costs
print "c2 costs:",c2_costs
plt.legend((line1, line2), ('c1', 'c2'))
plt.show()
