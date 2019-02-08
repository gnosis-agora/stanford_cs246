import re
import sys
from pyspark import SparkConf, SparkContext
from functools import partial
from itertools import combinations

conf = SparkConf()
sc = SparkContext(conf=conf)
lines = sc.textFile(sys.argv[1])
output_filepath = './output'
# def helper functions
def process_line(line):
	"""
	Finds all friend pairs
	"""
	collection = line.split('\t')
	friends = collection[1].split(',') if len(collection[1]) > 0 else []
	friends_int = [int(friend) for friend in friends]
	return (int(collection[0]),friends_int)

def get_friends_list(line):
	"""
	Returns the RDD that contains all friend pairs
	"""
	collection = line.split('\t')
	friends = collection[1].split(',') if len(collection[1]) > 0 else []
	return [(int(collection[0]), int(friend)) for friend in friends]

def get_top_recommendations(mututal_friend_list):
	"""
	Return the top 10 users based on mutual friends count
	"""
	# sort by top mutual friends desc
	# Then sort by friend id asc
	sorted_list = map(lambda friend: int(friend[0]),sorted(mututal_friend_list,key=lambda ele: (-ele[1],ele[0])))
	return sorted_list[:10]

friend_list = lines.flatMap(lambda l: get_friends_list(l))

recommentation_list = lines.map(lambda l : process_line(l)) \
.flatMapValues(partial(combinations, r=2)) \
.map(lambda (friend, pair) : pair) \
.subtract(friend_list) \
.map(lambda collection: (collection,1)) \
.reduceByKey(lambda c1, c2: c1 + c2) \
.flatMap(lambda (pair, count): ((pair[0],(pair[1],count)),(pair[1],(pair[0],count)))) \
.groupByKey().mapValues(list) \
.map(lambda (user,mutual_friend_list): (user,get_top_recommendations(mutual_friend_list))) \
.saveAsTextFile(output_filepath)

# print (recommentation_list.collect())
sc.stop()