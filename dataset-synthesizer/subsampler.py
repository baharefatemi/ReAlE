import os
import datetime
from random import seed
from random import randint
from random import random
import random
import argparse
import numpy as np
from itertools import permutations
from collections import defaultdict

DEFAULT_SAVE_DIR = './outputs'
seed(3)

class Subsampler:
    def __init__(self, tuples_per_rel, sub_sampling_p, valid_p, test_p, output_dir, max_arity):
    	self.sub_sampling_p = sub_sampling_p
    	self.tuples_per_rel = tuples_per_rel
    	self.knowledge_graph = defaultdict(lambda: [])
    	self.ents_in_train = defaultdict(lambda: False)
    	self.rels_in_train = defaultdict(lambda: False)
    	self.valid_p = valid_p
    	self.test_p = test_p

    	self.train_tuples = defaultdict(lambda: [])
    	self.valid_tuples = defaultdict(lambda: [])
    	self.test_tuples = defaultdict(lambda: [])

    	self.output_dir = output_dir

    	self.max_arity = max_arity

    def sub_sample(self):
    	for rel in self.tuples_per_rel:
    		for t in self.tuples_per_rel[rel]:
    			if len(t) <= self.max_arity:
	    			if random.random() < self.sub_sampling_p:
	    				self.knowledge_graph[rel].append(t)

    def decompose(self):
    	for rel in self.knowledge_graph:
    		for t in self.knowledge_graph[rel]:
    			rand_ = random.random()

    			if rand_ < (self.valid_p + self.test_p):
    				all_in_train = True
    				for ent in t:
    					if not self.ents_in_train[ent]:
    						all_in_train = False
    				if not self.rels_in_train[rel]:
    					all_in_train = False
    				if all_in_train:
    					if rand_ < (self.valid_p):
    						self.valid_tuples[rel].append(t)
    					else:
    						self.test_tuples[rel].append(t)
    			else:
    				self.train_tuples[rel].append(t)
    				self.rels_in_train[rel] = True
    				for ent in t:
    					self.ents_in_train[ent] = True

    def write_files(self):
    	self.write_files_(self.train_tuples, os.path.join(self.output_dir, 'train.txt'))
    	self.write_files_(self.valid_tuples, os.path.join(self.output_dir, 'valid.txt'))
    	self.write_files_(self.test_tuples, os.path.join(self.output_dir, 'test.txt'))

    def write_files_(self, tuples, output_dir):
    	output_file = open(output_dir, 'w')
    	tuples_rand = self.randomize(tuples)
    	for line in tuples_rand:
    		output_file.write('\t'.join([str(int(x)) for x in line]))
    		output_file.write('\n')
    	output_file.close()

    def randomize(self, tuples):
    	max_arity, n_tuples = self.stats(tuples)
    	tuples_np = np.zeros((n_tuples, max_arity + 1))

    	ind = 0
    	for rel in tuples:
    		for t in tuples[rel]:
    			tuples_np[ind][0] = rel
    			tuples_np[ind][1:len(t) + 1] = t
    			ind += 1
    	np.random.shuffle(tuples_np)
    	return tuples_np

    def stats(self, tuples):
    	max_arity = 0
    	n_tuples = 0
    	for rel in tuples:
    		for t in tuples[rel]:
    			max_arity = max(max_arity, len(t))
    			n_tuples += 1
    	return max_arity, n_tuples



						
