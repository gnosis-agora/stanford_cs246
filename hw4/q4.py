import math
import sys
import time
import matplotlib.pyplot as plt

def hash_fun(a, b, p, n_buckets, x):
	y = x % p
	hash_val = (a*y + b) % p
	return (hash_val % n_buckets)

def generate_hashes(hash_param_dir, p, n_buckets):
	"""
	Returns an array of hash functions that require parameter x and an empty dict 
	to keep track of count
	"""
	hashes = []
	hashes_count = []
	with open(hash_param_dir) as f:
		for line in f.readlines():
			a,b = line.strip().split('\t')
			hashes.append(lambda x: hash_fun(int(a),int(b),p,n_buckets,x))
			hashes_count.append({})
	return hashes, hashes_count

def get_estimated_count(hashes, hashes_count, x):
	"""
	Returns the estimated count of word x
	"""
	min_count = sys.maxint
	for i in range(len(hashes_count)):
		hash_val = hashes[i](x)
		min_count = min(min_count, hashes_count[i][hash_val])
	if min_count == sys.maxint:
		return 0
	return min_count


def compute_relative_error(fi, estimated_word_count):
	return float(estimated_word_count - fi) / fi

delta = math.e ** -5
epsilon = math.e * 10**-4
n_buckets = int(math.e / epsilon)
p = 123457

hash_param_dir = './q4/data/hash_params.txt'
words_stream_dir = './q4/data/words_stream.txt'
counts_dir = './q4/data/counts.txt'

hashes, hashes_count = generate_hashes(hash_param_dir, p, n_buckets)

start1 = time.clock()
with open(words_stream_dir) as f:
	t = 0
	for line in f.readlines():
		t += 1
		if t % (10 ** 6) == 0:
			print t, "records"
		x = int(line.strip())
		for idx in range(len(hashes)):
			hash_val = hashes[idx](x)
			if hash_val not in hashes_count[idx]:
				hashes_count[idx][hash_val] = 1
			else:
				hashes_count[idx][hash_val] += 1
end1 = time.clock()
print "time for counting:", (end1 - start1)

error = []
exact_word_frequency = []
start2 = time.clock()
with open(counts_dir) as f:
	count_num = 0
	for line in f.readlines():
	    count_num += 1
	    if count_num % (10 ** 4) == 0:
	        print count_num, "counts"
	    items = line.strip().split("\t")
	    word = int(items[0])
	    fi = int(items[1])
	    estimated_word_count = get_estimated_count(hashes, hashes_count, word)
	    error.append(compute_relative_error(fi, estimated_word_count))
	    exact_word_frequency.append(fi/float(t))
end2 = time.clock()
print "time for calculating error:", (end2 - start2)

plt.loglog(exact_word_frequency, error, "+")
plt.title("Relative Error vs Word Frequency")
plt.xlabel("Word Frequency (log)")
plt.ylabel("Relative Error (log)")
plt.grid()
plt.show()
