from pyspark import SparkConf, SparkContext
import numpy as np

conf = SparkConf()
sc = SparkContext(conf=conf)

file_dir = "./q2/data/test.txt"

def create_m(file_dir):
	entries = sc.textFile(file_dir) \
	.map(lambda l: l.strip().split('\t')) \
	.map(lambda (s,d): (int(s),int(d))) \
	.distinct() 

	outgoing_weights = entries.map(lambda (s,d): (s,1)).reduceByKey(lambda c1,c2: c1 + c2).collectAsMap()

	M = entries.map(lambda (s,d): ((d,s),float(1)/float(outgoing_weights[s])))
	return M.collectAsMap()

def update_r(n,beta,M,old_r):
	# apply M * old_r matrix multiplication here
	m_r = old_r
	for row in range(n):
		total = 0
		for col in range(n):
			if (row+1,col+1) in M:
				total += M[(row+1,col+1)] * m_r[col]
		m_r[row] = total
	new_r = [(1 - beta)/float(n) + beta * ele for ele in m_r] 
	return new_r

def create_l(file_dir):
	entries = sc.textFile(file_dir) \
	.map(lambda l: l.strip().split('\t')) \
	.map(lambda (s,d): (int(s),int(d))) \
	.distinct()

	return entries.collect()

def get_a(u,h, file_dir):
	L_entries = sc.textFile(file_dir) \
	.map(lambda l: l.strip().split('\t')) \
	.map(lambda (s,d): (int(s),int(d))) \
	.distinct() \
	.map(lambda (col, row): (row, h[col-1] * u)) \
	.reduceByKey(lambda c1,c2 : c1 + c2)

	L_entries_max = L_entries.max(lambda x: x[1])[1]
	new_a = L_entries.map(lambda (row, value): (row-1,float(value)/L_entries_max)) \
	.collectAsMap()
	return new_a

def get_h(lam, a, file_dir):
	L_entries = sc.textFile(file_dir) \
	.map(lambda l: l.strip().split('\t')) \
	.map(lambda (s,d): (int(s),int(d))) \
	.distinct() \
	.map(lambda (row, col): (row, a[col-1] * lam)) \
	.reduceByKey(lambda c1,c2 : c1 + c2)

	L_entries_max = L_entries.max(lambda x: x[1])[1]
	new_h = L_entries.map(lambda (row, value): (row-1,float(value)/L_entries_max)) \
	.collectAsMap()
	return new_h
# question 2a
n = 1000
beta = 0.8
M = create_m(file_dir)
r = [float(1)/n for i in range(n)]
for i in range(40):
	r = update_r(n,beta,M,r)

r = sorted([(idx+1, r[idx]) for idx in range(len(r))], key=lambda ele: -ele[1])
print "top 5: {0}".format([ele[0] for ele in r[:5]])
bot_5 = [ele[0] for ele in r[-5:]]
bot_5.reverse()
print "bottom 5: {0}".format(bot_5)

# question 2b
n = 4
u = 1
lam = 1
h = [1 for i in range(n)]
L = create_l(file_dir)
for i in range(40):
	a = get_a(u, h, file_dir)
	h = get_h(lam, a, file_dir)

print a
print h
a = sorted([(idx+1, a[idx]) for idx in range(len(a))], key=lambda ele: -ele[1])
h = sorted([(idx+1, h[idx]) for idx in range(len(h))], key=lambda ele: -ele[1])
print "A category"
print "top 5: {0}".format([ele[0] for ele in a[:5]])
a_bot_5 = [ele[0] for ele in a[-5:]]
a_bot_5.reverse()
print "bottom 5: {0}".format(a_bot_5)
print "H category"
print "top 5: {0}".format([ele[0] for ele in h[:5]])
h_bot_5 = [ele[0] for ele in h[-5:]]
h_bot_5.reverse()
print "bottom 5: {0}".format(h_bot_5)









