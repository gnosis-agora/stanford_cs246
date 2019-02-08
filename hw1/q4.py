# Authors: Jessica Su, Wanzi Zhou, Pratyaksh Sharma, Dylan Liu, Ansh Shukla

import numpy as np
import random
import time
import pdb
import unittest
from PIL import Image
import matplotlib.pyplot as plt

# Finds the L1 distance between two vectors
# u and v are 1-dimensional np.array objects
# TODO: Implement this
def l1(u, v):
    return np.sum(abs(u - v))

# Loads the data into a np array, where each row corresponds to
# an image patch -- this step is sort of slow.
# Each row in the data is an image, and there are 400 columns.
def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')

# Creates a hash function from a list of dimensions and thresholds.
def create_function(dimensions, thresholds):
    def f(v):
        boolarray = [v[dimensions[i]] >= thresholds[i] for i in range(len(dimensions))]
        return "".join(map(str, map(int, boolarray)))
    return f

# Creates the LSH functions (functions that compute L K-bit hash keys).
# Each function selects k dimensions (i.e. column indices of the image matrix)
# at random, and then chooses a random threshold for each dimension, between 0 and
# 255.  For any image, if its value on a given dimension is greater than or equal to
# the randomly chosen threshold, we set that bit to 1.  Each hash function returns
# a length-k bit string of the form "0101010001101001...", and the L hash functions 
# will produce L such bit strings for each image.
def create_functions(k, L, num_dimensions=400, min_threshold=0, max_threshold=255):
    functions = []
    for i in range(L):
        dimensions = np.random.randint(low = 0, 
                                   high = num_dimensions,
                                   size = k)
        thresholds = np.random.randint(low = min_threshold, 
                                   high = max_threshold + 1, 
                                   size = k)

        functions.append(create_function(dimensions, thresholds))
    return functions

# Hashes an individual vector (i.e. image).  This produces an array with L
# entries, where each entry is a string of k bits.
def hash_vector(functions, v):
    return np.array([f(v) for f in functions])

# Hashes the data in A, where each row is a datapoint, using the L
# functions in "functions."
def hash_data(functions, A):
    return np.array(list(map(lambda v: hash_vector(functions, v), A)))

# Retrieve all of the points that hash to one of the same buckets 
# as the query point.  Do not do any random sampling (unlike what the first
# part of this problem prescribes).
# Don't retrieve a point if it is the same point as the query point.
def get_candidates(hashed_A, hashed_point, query_index):
    return filter(lambda i: i != query_index and \
        any(hashed_point == hashed_A[i]), range(len(hashed_A)))

# Sets up the LSH.  You should try to call this function as few times as 
# possible, since it is expensive.
# A: The dataset.
# Return the LSH functions and hashed data structure.
def lsh_setup(A, k = 24, L = 10):
    functions = create_functions(k = k, L = L)
    hashed_A = hash_data(functions, A)
    return (functions, hashed_A)

# Run the entire LSH algorithm
def lsh_search(A, hashed_A, functions, query_index, num_neighbors = 10):
    hashed_point = hash_vector(functions, A[query_index, :])
    candidate_row_nums = get_candidates(hashed_A, hashed_point, query_index)
    
    distances = map(lambda r: (r, l1(A[r], A[query_index])), candidate_row_nums)
    best_neighbors = sorted(distances, key=lambda t: t[1])[:num_neighbors]

    return [t[0] for t in best_neighbors]

# Plots images at the specified rows and saves them each to files.
def plot(A, row_nums, base_filename):
    for row_num in row_nums:
        patch = np.reshape(A[row_num, :], [20, 20])
        im = Image.fromarray(patch)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(base_filename + "-" + str(row_num) + ".png")

# Finds the nearest neighbors to a given vector, using linear search.
def linear_search(A, query_index, num_neighbors):
    neighbors_arr = []
    vector = A[query_index]
    for row_index in range(len(A)):
        if row_index != query_index:
            candidate = A[row_index]
            distance = l1(candidate,vector)
            if len(neighbors_arr) < num_neighbors:
                neighbors_arr.append([row_index,distance])
                max_neighbor = max(neighbors_arr,key=lambda x: x[1])
                neighbors_arr = sorted(neighbors_arr, key=lambda x: x[1])
            else:
                if distance > neighbors_arr[-1][1]:
                    continue
                candidate_idx = -1
                for idx in range(len(neighbors_arr),0,-1):
                    if neighbors_arr[idx-1][1] > distance:
                        candidate_idx = idx-1
                neighbors_arr.pop(candidate_idx)
                neighbors_arr.insert(candidate_idx,[row_index,distance])
    sorted_arr = sorted(neighbors_arr, key=lambda x: x[1])
    return [neighbor[0] for neighbor in sorted_arr]

# TODO: Write a function that computes the error measure
def get_error(A, image_patches, linear_neighbors, lsh_neighbours):
    error_sum = 0
    for j in range(10):
        patch = image_patches[j]
        linear_distance = sum([l1(A[idx],A[patch]) for idx in linear_neighbors[j]])
        lsh_distance = sum([l1(A[idx],A[patch]) for idx in lsh_neighbours[j]])
        error_sum += lsh_distance/linear_distance
    return error_sum/10

# TODO: Solve Problem 4
def run_simulation(data,K,L):
    linear_neighbors = []
    lsh_neighbours = []
    functions, hashed_A = lsh_setup(data,K,L)
    for i in [99,199,299,399,499,599,699,799,899,999]:
        linear_neighbors.append(linear_search(data, i, 3))
        lsh_neighbours.append(lsh_search(data,hashed_A,functions,i,3))
    return linear_neighbors,lsh_neighbours

def problem4():
    start_time = time.time()
    data = load_data("./data/patches.csv")
    print("data loaded in %s seconds" % (time.time() - start_time))

    # Test average search time
    functions, hashed_A = lsh_setup(data)
    linear_time = 0.0
    lsh_time = 0.0
    for i in [99,199,299,399,499,599,699,799,899,999]:
        print("row: %s"%i)
        start_time = time.time()
        neighbors = linear_search(data, i, 3)
        run_time = time.time() - start_time
        print("neighbors: {0} found in {1} seconds".format(neighbors,run_time))
        linear_time += run_time

        start_time = time.time()
        neighbors = lsh_search(data,hashed_A,functions,i,3)
        run_time = time.time() - start_time
        print("neighbors: {0} found in {1} seconds".format(neighbors,run_time))
        lsh_time += run_time
    print('')
    print("linear average: {0}".format(linear_time/10))
    print("lsh average: {0}".format(lsh_time/10))      

    # Plot error to L
    image_patches = [99,199,299,399,499,599,699,799,899,999]
    # error_arr = []
    
    # for L in range(10,22,2):
    #     linear_neighbors,lsh_neighbours = run_simulation(data,24,L)
    #     error_arr.append(get_error(data, image_patches, linear_neighbors, lsh_neighbours))
    # L = [l for l in range(10,22,2)]
    # plt.clf() 
    # plt.close()
    # plt.plot(L,error_arr)
    # plt.xlabel('L')
    # plt.ylabel('error')
    # plt.savefig('L_error.png')

    # Plot error to k
    error_arr = []
    for k in range(16,26,2):
        linear_neighbors,lsh_neighbours = run_simulation(data,k,10)
        error_arr.append(get_error(data, image_patches, linear_neighbors, lsh_neighbours))
    K = [k for k in range(16,26,2)]
    print K
    print error_arr
    plt.plot(K,error_arr)
    plt.xlabel('K')
    plt.ylabel('error')
    plt.savefig('K_error.png')

#### TESTS #####

class TestLSH(unittest.TestCase):
    def test_l1(self):
        u = np.array([1, 2, 3, 4])
        v = np.array([2, 3, 2, 3])
        self.assertEqual(l1(u, v), 4)

    def test_hash_data(self):
        f1 = lambda v: sum(v)
        f2 = lambda v: sum([x * x for x in v])
        A = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(f1(A[0,:]), 6)
        self.assertEqual(f2(A[0,:]), 14)

        functions = [f1, f2]
        self.assertTrue(np.array_equal(hash_vector(functions, A[0, :]), np.array([6, 14])))
        self.assertTrue(np.array_equal(hash_data(functions, A), np.array([[6, 14], [15, 77]])))

    ### TODO: Write your tests here (they won't be graded, 
    ### but you may find them helpful)

if __name__ == '__main__':
    # unittest.main() ### TODO: Uncomment this to run tests
    problem4()
    # start_time = time.time()
    # data = load_data("./data/patches.csv")
    # print("data loaded in %s seconds" % (time.time() - start_time))