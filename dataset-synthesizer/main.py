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
from subsampler import Subsampler
import json
import sys

DEFAULT_SAVE_DIR = './outputs'
seed(3)

class Synthesizer:
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.number_ent = args.number_ent
        self.number_rel = args.number_rel
        self.number_rel_all = self.number_rel
        self.number_edge = args.number_edge
        self.max_arity = args.max_arity
        self.min_arity = args.min_arity
        self.p_rename = args.p_rename
        self.p_projection = args.p_projection
        self.p_union = args.p_union
        self.p_product = args.p_product
        self.p_selection = args.p_selection
        self.p_setd = args.p_setd
        self.number_of_ops = args.number_of_ops


        self.tuples_per_rel = defaultdict(lambda: [])
        self.rel_per_arity = defaultdict(lambda: [])
        self.degree = {}
        self.ops_on_rel = {}
        self.output_dir = self.create_output_dir(args.output_dir)
        self.arities = self.create_arities()
        
        self.operations_log = open(os.path.join(self.output_dir, 'ops-logs.txt'), 'w')
        self.ground_truth_log = open(os.path.join(self.output_dir, 'ground-truth.txt'), 'w')
        print("ground truth generator in progress")
        self.init_tuples = self.create_init_graph()

        print("applying operations in progress")
        self.apply_operations()

        print("max degree", max(self.degree.values()))

        # self.load_tuples_per_rel()

    def create_output_dir(self, output_dir):
        """
        If an output dir is given, then make sure it exists. Otherwise, create one based on time stamp.
        """
        if output_dir is None:
            time = datetime.datetime.now()
            dataset_name = '{}_{}'.format(self.dataset_name, time.strftime("%Y%m%d-%H%M%S"))
            output_dir = os.path.join(DEFAULT_SAVE_DIR, self.dataset_name, dataset_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print("Created output directory {}".format(output_dir))
        return output_dir

    def create_arities(self):
        arities = {}
        for rel in range(self.number_rel):
            arity = randint(self.min_arity, self.max_arity)
            arities[rel] = arity
            self.degree[rel] = 0
        return arities

    def rel_per_arity_init(self):
        for rel, arity in enumerate(self.arities):
            self.rel_per_arity[self.arities[rel]].append(rel)

    def create_init_graph(self):
        for edge in range(self.number_edge):
            rel = randint(0, self.number_rel-1)
            entities = []
            arity = self.arities[rel]
            for ar in range(arity):
                entities.append(randint(1, self.number_ent))
            self.tuples_per_rel[rel].append(entities)

        self.rel_per_arity_init()

        for r in self.tuples_per_rel:
            self.tuples_per_rel[r] = np.array(self.tuples_per_rel[r])

    def add_tuples_per_rel(self, tuples, rel):

        # print("1", "%d bytes" % (sys.getsizeof(self.tuples_per_rel)))
        # print("2", "%d bytes" % (sys.getsizeof(np.array(tuples).astype(int))))
        self.tuples_per_rel[rel] = np.array(tuples).astype(int)
        # print(len(self.tuples_per_rel))
        # for t in tuples:
        #     self.tuples_per_rel[rel].append(np.array(t))

    def such_arity_exist(self, arity):
        if len(self.rel_per_arity[arity]) > 1:
            return True
        return False

    def apply_operations(self):
        self.operations_log.write("Applying operations" + '\n')
        print("Applying operations")
        ind_operation = 0

        while ind_operation < args.number_of_ops:
            print("op", ind_operation)

            op = random.choices(np.arange(1, 7), weights=[args.p_rename, args.p_projection, args.p_union, args.p_product, args.p_selection, args.p_setd])[0]
            ind_operation += 1
            rel1 = randint(0, self.number_rel_all - 1)

            if op == 1:
                self.rename(rel1)
            elif op == 2:
                self.projection(rel1)
            elif op == 3:
                such_arity_exist = self.such_arity_exist(self.arities[rel1])
                if such_arity_exist:
                    rel2 = random.choice(self.rel_per_arity[self.arities[rel1]])
                    while rel1 == rel2:
                        rel2 = random.choice(self.rel_per_arity[self.arities[rel1]])
                    self.union(rel1, rel2)
            elif op == 4:
                rel2 = randint(0, self.number_rel_all - 1)
                while rel1 == rel2:
                    rel2 = randint(0, self.number_rel_all - 1)
                self.product(rel1, rel2)
            elif op == 5:
                while self.arities[rel1] < 2:
                    rel1 = randint(0, self.number_rel_all - 1)
                self.selection(rel1)
            else:
                such_arity_exist = self.such_arity_exist(self.arities[rel1])
                if such_arity_exist:
                    rel2 = random.choice(self.rel_per_arity[self.arities[rel1]])
                    while rel1 == rel2:
                        rel2 = random.choice(self.rel_per_arity[self.arities[rel1]])
                    self.setd(rel1, rel2)


        json_degree = json.dumps(self.degree)
        json_ops = json.dumps(self.ops_on_rel)
        self.operations_log.write(json_degree)
        self.operations_log.write("\n")
        self.operations_log.write(json_ops)
        self.operations_log.close()

        for rel in self.tuples_per_rel:
            json_rel = json.dumps(rel) 
            json_tuples = json.dumps(self.tuples_per_rel[rel].tolist())
            self.ground_truth_log.write(json_rel)
            self.ground_truth_log.write('\n')
            self.ground_truth_log.write(json_tuples)
            self.ground_truth_log.write('\n')
        self.ground_truth_log.close()



    def create_new_permutation(self, arity):
        all_permutations = list(permutations(range(0, arity)))
        return random.choice(all_permutations)

    def rename(self, rel):
        self.operations_log.write("Rename operation for relation {}".format(str(rel)) + '\n')
        print("Rename operation for relation {}".format(str(rel)))
        curr_tuples = self.tuples_per_rel[rel]
        
        new_permutation = self.create_new_permutation(self.arities[rel])
        new_rel = self.number_rel_all
        self.degree[new_rel] = self.degree[rel] + 1
        print("degree", self.degree[new_rel])
        self.ops_on_rel[new_rel] = 'rename'
        self.operations_log.write("New relation id {}".format(str(new_rel)) + '\n')
        print("New relation id {}".format(str(new_rel)))
        rename_tuples = np.zeros((len(curr_tuples),  self.arities[rel]))
        self.number_rel_all += 1

        for ind, p in enumerate(new_permutation):
            rename_tuples[:, ind] = curr_tuples[:, p]

        self.sanity_check(rename_tuples, self.arities[rel], new_rel)


    def projection(self, rel):
        self.operations_log.write("Projection operation for relation {}".format(str(rel)) + '\n')
        print("Projection operation for relation {}".format(str(rel)))
        curr_tuples = self.tuples_per_rel[rel]
        new_rel_arity = randint(1, self.arities[rel])
        projection_tuples = np.zeros((len(curr_tuples),  new_rel_arity))
        new_rel = self.number_rel_all
        self.degree[new_rel] = self.degree[rel] + 1
        print("degree", self.degree[new_rel])
        self.ops_on_rel[new_rel] = 'project'
        self.operations_log.write("New relation id {}".format(str(new_rel)) + '\n')
        print("New relation id {}".format(str(new_rel)))
        self.number_rel_all += 1

        permutation = random.sample(range(0, self.arities[rel]), new_rel_arity)

        for ind, p in enumerate(permutation):
            projection_tuples[:, ind] = curr_tuples[:, p]
        self.sanity_check(projection_tuples, new_rel_arity, new_rel)


    def union(self, rel1, rel2):
        self.operations_log.write("Union operation for relation {} and {}".format(str(rel1), str(rel2)) + '\n')
        print("Union operation for relation {} and {}".format(str(rel1), str(rel2)))
        curr_tuples1 = self.tuples_per_rel[rel1]
        curr_tuples2 = self.tuples_per_rel[rel2]
        new_rel = self.number_rel_all
        self.degree[new_rel] = max(self.degree[rel1], self.degree[rel2]) + 1
        print("degree", self.degree[new_rel])
        self.ops_on_rel[new_rel] = 'union'
        self.operations_log.write("New relation id {}".format(str(new_rel)) + '\n')
        print("New relation id {}".format(str(new_rel)))
        self.number_rel_all += 1
        union_tuples = np.concatenate((curr_tuples1, curr_tuples2))
        self.sanity_check(union_tuples, self.arities[rel1], new_rel)


    def product(self, rel1, rel2):
        

        if self.arities[rel1] + self.arities[rel2] <= self.max_arity:
            self.operations_log.write("Product operation for relation {} and {}".format(str(rel1), str(rel2)) + '\n')
            print("Product operation for relation {} and {}".format(str(rel1), str(rel2)))
            curr_tuples1 = self.tuples_per_rel[rel1]
            curr_tuples2 = self.tuples_per_rel[rel2]
            new_rel = self.number_rel_all
            self.degree[new_rel] = max(self.degree[rel1], self.degree[rel2]) + 1
            print("degree", self.degree[new_rel])
            self.ops_on_rel[new_rel] = 'product'
            product_tuples = np.zeros((len(curr_tuples1) * len(curr_tuples2),  self.arities[rel1] + self.arities[rel2]))
            
            self.operations_log.write("New relation id {}".format(str(new_rel)) + '\n')
            print("New relation id {}".format(str(new_rel)))
            self.number_rel_all += 1

            repeated1 = np.repeat(curr_tuples1, curr_tuples2.shape[0], 0)
            repeated2 = np.tile(curr_tuples2, (curr_tuples1.shape[0], 1))
            product_tuples = np.concatenate((repeated1, repeated2), 1)

            self.sanity_check(product_tuples, self.arities[rel1] + self.arities[rel2], new_rel)


    def selection(self, rel):
        self.operations_log.write("Selection operation for relation {}".format(str(rel)) + '\n')
        print("Selection operation for relation {}".format(str(rel)))
        curr_tuples = self.tuples_per_rel[rel]
        new_rel = self.number_rel_all
        self.degree[new_rel] = self.degree[rel] + 1
        print("degree", self.degree[new_rel])
        self.ops_on_rel[new_rel] = 'select'
        self.operations_log.write("New relation id {}".format(str(new_rel)) + '\n')
        print("New relation id {}".format(str(new_rel)))
        # selection_tuples = np.zeros((len(curr_tuples),  self.arities[rel] + 1))
        
        self.number_rel_all += 1

        selection_type = random.random()
        if selection_type < 0.5:
            pos1, pos2 = random.sample(range(0, self.arities[rel]), 2)
            selection_tuples = curr_tuples[curr_tuples[:, pos1] == curr_tuples[:, pos2]]

        else:
            pos1 = randint(0, len(curr_tuples[0]) - 1)
            c = randint(0, self.number_ent - 1)
            selection_tuples = curr_tuples[curr_tuples[:, pos1] == c]


        selection_tuples = self.remove_one_column(selection_tuples, pos1)

        self.sanity_check(selection_tuples, self.arities[rel] - 1, new_rel)



    def remove_one_column(self, tuples, col):
        tuples[:, col] = np.zeros(tuples.shape[0])
        tuples[:, col:] = np.roll(tuples[:,col:], -1, axis=1)
        return tuples[:, :-1]

    def setd(self, rel1, rel2):
        self.operations_log.write("Setd operation for relation {} and {}".format(str(rel1), str(rel2)) + '\n')
        print("Setd operation for relation {} and {}".format(str(rel1), str(rel2)) )
        curr_tuples1 = self.tuples_per_rel[rel1]
        curr_tuples2 = self.tuples_per_rel[rel2]
        
        new_rel = self.number_rel_all
        self.degree[new_rel] = max(self.degree[rel1], self.degree[rel2]) + 1
        print("degree", self.degree[new_rel])
        self.ops_on_rel[new_rel] = 'setd'
        self.operations_log.write("New relation id {}".format(str(new_rel)) + '\n')
        print("New relation id {}".format(str(new_rel)))
        
        self.number_rel_all += 1


        rows_1 = curr_tuples1.view([('', curr_tuples1.dtype)] * curr_tuples1.shape[1])
        # print("3"rows_1)
        rows_2 = curr_tuples2.view([('', curr_tuples2.dtype)] * curr_tuples2.shape[1])
        setd_tuples = np.setdiff1d(rows_1, rows_2).view(curr_tuples1.dtype).reshape(-1, curr_tuples1.shape[1])
        self.sanity_check(setd_tuples, self.arities[rel1], new_rel)

        
    def sanity_check(self, tuples, arity, rel_id):
        if len(tuples) > 0:
            self.add_tuples_per_rel(tuples, rel_id)
            self.arities[rel_id] = arity
            self.rel_per_arity[arity].append(rel_id)
        else:
            self.number_rel_all -= 1
            self.operations_log.write("Empty relation is created and deleted" + '\n')
            print("Empty relation is created and deleted")

    # def load_tuples_per_rel(self):

    #     ground_truth_log = open(os.path.join('outputs', 'Small', 'Small_20200430-135407', 'ground-truth.txt'), 'r')

    #     curr_rel = 0
    #     for ind, line in enumerate(ground_truth_log):
    #         if ind % 2 == 0:
    #             curr_rel = int(line)
    #         else:
    #             b_new = json.loads(line)
    #             a_new = np.array(b_new)
    #             self.tuples_per_rel[curr_rel] = a_new

        # for rel in self.tuples_per_rel:
        #     json_rel = json.dumps(rel) 
        #     json_tuples = json.dumps(self.tuples_per_rel[rel].tolist())
        #     self.ground_truth_log.write(json_rel)
        #     self.ground_truth_log.write('\n')
        #     self.ground_truth_log.write(json_tuples)
        #     self.ground_truth_log.write('\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-number_ent', type=int, default=10)
    parser.add_argument('-number_rel', type=int, default=5)
    parser.add_argument('-number_edge', type=int, default=50)
    parser.add_argument('-max_arity', type=int, default=3)
    parser.add_argument('-min_arity', type=int, default=2)
    parser.add_argument('-p_rename', type=float, default=0.10)
    parser.add_argument('-p_projection', type=float, default=0.10)
    parser.add_argument('-p_union', type=float, default=0.10)
    parser.add_argument('-p_product', type=float, default=0.10)
    parser.add_argument('-p_selection', type=float, default=0.10)
    parser.add_argument('-p_setd', type=float, default=0.10)
    parser.add_argument('-number_of_ops', type=int, default=10)
    parser.add_argument('-output_dir', type=str, default=None, help="A path to the directory where the dataset will be saved and/or loaded from.")
    parser.add_argument('-dataset_name', type=str, default="Small")
    parser.add_argument('-sub_sampling_p', type=float, default=0.50)
    parser.add_argument('-valid_p', type=float, default=0.10)
    parser.add_argument('-test_p', type=float, default=0.10)
    
    args = parser.parse_args()

    synthesizer = Synthesizer(args)
    subsampler = Subsampler(synthesizer.tuples_per_rel, args.sub_sampling_p, args.valid_p, args.test_p, synthesizer.output_dir, args.max_arity)
    print("sub sampling in progress")
    subsampler.sub_sample()
    print("decomposing in progress")
    subsampler.decompose()
    print("write files in progress")
    subsampler.write_files()
