import re
import sys
from pyspark import SparkConf, SparkContext
from functools import partial
from itertools import combinations

conf = SparkConf()
sc = SparkContext(conf=conf)
lines = sc.textFile(sys.argv[1])
output_filepath = './q2_output'
# def helper functions
def normalize_group(*args):
    return (sorted(args))

def get_pairs(items, frequent_items):
	pairs = []
	filtered_items = [item for item in items if item in frequent_items]
	for idx1 in range(len(filtered_items)-1):
		for idx2 in range(idx1+1,len(filtered_items)):
			pairs.append(normalize_group(filtered_items[idx1],filtered_items[idx2]))
	return pairs
def common_elements(list1, list2):
    return list(set(list1) & set(list2))

def get_triple(lists):
	set = []
	for list in lists:
		for ele in list:
			if ele not in set:
				set.append(ele)
	return tuple(sorted(set)) if len(set) > 0 else ()

frequent_items = lines.flatMap(lambda l : l.split()) \
.map(lambda ele: (ele,1)) \
.reduceByKey(lambda e1,e2: e1 + e2) \
.filter(lambda x: x[1] >= 100)


frequent_itemset = frequent_items.collectAsMap()

frequent_pairs = lines.map(lambda l: l.split()) \
.flatMap(lambda l: get_pairs(l,frequent_itemset)) \
.map(lambda pair: (tuple(pair),1)) \
.reduceByKey(lambda p1, p2: p1 + p2) \
.filter(lambda x: x[1] >= 100) 

freq_pairs = frequent_pairs.map(lambda ele: list(ele[0])).collect()
freq_pairs_with_count = frequent_pairs.collectAsMap()

freq_pair_conf = frequent_pairs.flatMap(lambda (pairs, count): (((pairs[0],pairs[1]),count), ((pairs[1],pairs[0]), count))) \
.map(lambda (pair,count): (pair, float(count*1000/frequent_itemset[pair[0]]))) \
.map(lambda (pair, conf): (pair, conf/1000)) \
.sortBy(lambda x: -x[1]) 

print (freq_pair_conf.take(40))
# print frequent_itemset
sc.stop()