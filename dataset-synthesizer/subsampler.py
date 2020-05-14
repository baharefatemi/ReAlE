import os
import datetime
import random
import numpy as np
from collections import defaultdict

DEFAULT_SAVE_DIR = './outputs'
random.seed(3)
np.random.seed(3)

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
				if len(t) <= self.max_arity and len(t) >= 2:
					if random.random() < self.sub_sampling_p:
						self.knowledge_graph[rel].append(t)
		del self.tuples_per_rel

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
		del self.knowledge_graph


	def write_files(self):
		self.write_files_(self.train_tuples, os.path.join(self.output_dir, 'train.txt'))
		del self.train_tuples
		self.write_files_(self.valid_tuples, os.path.join(self.output_dir, 'valid.txt'))
		del self.valid_tuples
		self.write_files_(self.test_tuples, os.path.join(self.output_dir, 'test.txt'))
		del self.test_tuples

	def write_files_(self, tuples, output_dir):
		output_file = open(output_dir, 'w')
		# tuples_rand = self.randomize(tuples)

		for rel in tuples:
			for t in tuples[rel]:
				output_file.write(str(rel) + '\t')
				output_file.write('\t'.join([str(int(x)) for x in t]))
				output_file.write('\n')
		output_file.close()

	def randomize(self, tuples):
		print(len(tuples))
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


	def decompose_test_file_operation(self, ops_on_rel):
		input_file = open(os.path.join(self.output_dir, 'test.txt'), 'r')
		output_dir = os.path.join(self.output_dir, 'operations')
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		operation_files = {
			'rename': open(os.path.join(output_dir, 'rename.txt'), 'w'),
			'project': open(os.path.join(output_dir, 'project.txt'), 'w'),
			'product': open(os.path.join(output_dir, 'product.txt'), 'w'),
			'union': open(os.path.join(output_dir, 'union.txt'), 'w'),
			'setd': open(os.path.join(output_dir, 'setd.txt'), 'w'),
			'select': open(os.path.join(output_dir, 'select.txt'), 'w'),
			'primitive': open(os.path.join(output_dir, 'primitive.txt'), 'w')}

		for line in input_file:
			rel = int(line.strip().split('\t')[0])
			if rel in ops_on_rel:
				operation_files[ops_on_rel[rel]].write(line)
			else:
				operation_files["primitive"].write(line)

	def decompose_test_file_degree(self, degree):
		input_file = open(os.path.join(self.output_dir, 'test.txt'), 'r')
		output_dir = os.path.join(self.output_dir, 'degrees')

		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
			
		degree_list = list(degree.values())
		degree_files = {}

		for deg in degree_list:
			degree_files[deg] = open(os.path.join(output_dir, str(deg) + '.txt'), 'w')

		for line in input_file:
			rel = int(line.strip().split('\t')[0])
			degree_files[degree[rel]].write(line)


	def decompose_test_file(self, ops_on_rel, degree):
		self.decompose_test_file_operation(ops_on_rel)
		self.decompose_test_file_degree(degree)
