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
# seed(3)

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

        self.tuples_per_rel = defaultdict(lambda: [])
        self.degree_rel = defaultdict(lambda: 0)
        self.rel_per_arity = defaultdict(lambda: [])
        self.degree = {}
        self.output_dir = self.create_output_dir(args.output_dir)
        self.arities = self.create_arities()
        
        self.init_tuples = self.create_init_graph()
        self.apply_operations()


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


    def add_tuples_per_rel(self, tuples):
        for t in tuples:
            self.tuples_per_rel[t[0]].append(list(t[1:]))

    def such_arity_exist(self, arity):
        if len(self.rel_per_arity[arity]) > 1:
            return True
        return False

    def apply_operations(self):
        print("Applying operations")

        total_operations = 10
        ind_operation = 0

        while ind_operation < total_operations:
            
            op = random.choices(np.arange(1, 7), weights=[args.p_rename, args.p_projection, args.p_union, args.p_product, args.p_selection, args.p_setd])[0]
            ind_operation += 1
            # op = randint(3, 3)
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
            elif op == 6:
                such_arity_exist = self.such_arity_exist(self.arities[rel1])
                if such_arity_exist:
                    rel2 = random.choice(self.rel_per_arity[self.arities[rel1]])
                    while rel1 == rel2:
                        rel2 = random.choice(self.rel_per_arity[self.arities[rel1]])
                    self.setd(rel1, rel2)


    def create_new_permutation(self, arity):
        all_permutations = list(permutations(range(0, arity)))
        return random.choice(all_permutations)

    def rename(self, rel):
        print("Rename operation for relation {}".format(str(rel)))
        curr_tuples = self.tuples_per_rel[rel]
        
        new_permutation = self.create_new_permutation(self.arities[rel])
        new_rel = self.number_rel_all
        self.degree[new_rel] = self.degree[rel] + 1
        print("New relation id {}".format(str(new_rel)))
        
        rename_tuples = np.zeros((len(curr_tuples),  self.arities[rel] + 1))
        
        self.number_rel_all += 1
        for ind, t in enumerate(curr_tuples):
            entities = []
            for p in new_permutation:
                entities.append(t[p])
            rename_tuples[ind][0] = new_rel
            rename_tuples[ind][1:self.arities[rel]+1] =  entities

        self.sanity_check(rename_tuples, self.arities[rel])


    def projection(self, rel):
        print("Projection operation for relation {}".format(str(rel)))
        curr_tuples = self.tuples_per_rel[rel]
        new_rel_arity = randint(1, self.arities[rel])
        projection_tuples = np.zeros((len(curr_tuples),  new_rel_arity + 1))
        new_rel = self.number_rel_all
        self.degree[new_rel] = self.degree[rel] + 1
        print("New relation id {}".format(str(new_rel)))
        
        self.number_rel_all += 1
        permutation = []
        while len(permutation) != new_rel_arity:
            ent_pos = randint(0, self.arities[rel] - 1)
            if ent_pos not in permutation:
                permutation.append(ent_pos)
        for ind, t in enumerate(curr_tuples):
            entities = []
            for p in permutation:
                entities.append(t[p])
            projection_tuples[ind][0] = new_rel
            projection_tuples[ind][1:new_rel_arity+1] =  entities

        self.sanity_check(projection_tuples, new_rel_arity)


    def union(self, rel1, rel2):
        print("Union operation for relation {} and {}".format(str(rel1), str(rel2)))
        curr_tuples1 = self.tuples_per_rel[rel1]
        curr_tuples2 = self.tuples_per_rel[rel2]
        new_rel = self.number_rel_all
        self.degree[new_rel] = max(self.degree[rel1], self.degree[rel2]) + 1
        union_tuples = np.zeros((len(curr_tuples1) + len(curr_tuples2),  self.arities[rel1] + 1))
        print("New relation id {}".format(str(new_rel)))
        self.number_rel_all += 1

        for ind, t in enumerate(curr_tuples1):
            union_tuples[ind][1:] = t
            union_tuples[ind][0] = new_rel
        for ind, t in enumerate(curr_tuples2):
            union_tuples[len(curr_tuples1) + ind][1:] = t
            union_tuples[len(curr_tuples1) + ind][0] = new_rel

        self.sanity_check(union_tuples, self.arities[rel1])


    def product(self, rel1, rel2):
        print("Product operation for relation {} and {}".format(str(rel1), str(rel2)))
        curr_tuples1 = self.tuples_per_rel[rel1]
        curr_tuples2 = self.tuples_per_rel[rel2]
        new_rel = self.number_rel_all
        self.degree[new_rel] = max(self.degree[rel1], self.degree[rel2]) + 1
        product_tuples = np.zeros((len(curr_tuples1) * len(curr_tuples2),  self.arities[rel1] + self.arities[rel2] + 1))
        
        print("New relation id {}".format(str(new_rel)))
        self.number_rel_all += 1
        for ind1, t1 in enumerate(curr_tuples1):
            for ind2, t2 in enumerate(curr_tuples2):
                product_tuples[(ind1 * len(curr_tuples2))+ ind2][0] = new_rel
                product_tuples[(ind1 * len(curr_tuples2))+ ind2][1:self.arities[rel1]+1] = t1[0:self.arities[rel1]]
                product_tuples[(ind1 * len(curr_tuples2))+ ind2][self.arities[rel1]+1:self.arities[rel1] + self.arities[rel2]+1] = t2[0:self.arities[rel2]]

        self.sanity_check(product_tuples, self.arities[rel1] + self.arities[rel2])

    def selection(self, rel):
        print("Selection operation for relation {}".format(str(rel)))
        curr_tuples = self.tuples_per_rel[rel]
        new_rel = self.number_rel_all
        self.degree[new_rel] = self.degree[rel] + 1
        print("New relation id {}".format(str(new_rel)))
        selection_tuples = np.zeros((len(curr_tuples),  self.arities[rel] + 1))
        
        self.number_rel_all += 1

        selection_type = random.random()
        if selection_type < 0.5:
            which_pos_1 = 0
            which_pos_2 = 0
            while which_pos_1 == which_pos_2:
                which_pos_1 = randint(0, self.arities[rel] - 1)
                which_pos_2 = randint(0, self.arities[rel] - 1)
            for ind, t in enumerate(curr_tuples):
                if t[which_pos_1] == t[which_pos_2]:
                    selection_tuples[ind][0] = new_rel
                    selection_tuples[ind][1:] = t
        else:
            which_pos_1 = randint(0, len(curr_tuples[0]) - 1)
            which_val = randint(0, self.number_ent - 1)
            for ind, t in enumerate(curr_tuples):
                if t[which_pos_1] == which_val:
                    selection_tuples[ind][0] = new_rel
                    selection_tuples[ind][1:] = t

        selection_tuples = self.remove_one_column(selection_tuples, which_pos_1 + 1)
        selection_tuples = self.remove_padding(selection_tuples)

        self.sanity_check(selection_tuples, self.arities[rel] - 1)


    def remove_one_column(self, tuples, col):
        tuples[:, col] = np.zeros(tuples.shape[0])
        tuples[:, col:] = np.roll(tuples[:,col:], -1, axis=1)
        return tuples[:, :-1]

    def setd(self, rel1, rel2):
        print("Setd operation for relation {} and {}".format(str(rel1), str(rel2)))
        curr_tuples1 = self.tuples_per_rel[rel1]
        curr_tuples2 = self.tuples_per_rel[rel2]
        
        new_rel = self.number_rel_all
        self.degree[new_rel] = max(self.degree[rel1], self.degree[rel2]) + 1
        print("New relation id {}".format(str(new_rel)))
        
        setd_tuples = np.zeros((len(curr_tuples1),  self.arities[rel1] + 1))
        
        self.number_rel_all += 1

        for ind, t in enumerate(curr_tuples1):
            curr_tuples2 = np.array(curr_tuples2)
            if not(tuple(np.array(t)) in {tuple(v): True for v in curr_tuples2}):
                setd_tuples[ind][1:] = t
                setd_tuples[ind][0] = new_rel
 
        setd_tuples = self.remove_padding(setd_tuples)
        self.sanity_check(setd_tuples, self.arities[rel1])

        
    def sanity_check(self, tuples, arity):
        if len(tuples) > 0:
            rel_id = tuples[0, 0]
            self.add_tuples_per_rel(tuples)
            self.arities[rel_id] = arity
            self.rel_per_arity[arity].append(rel_id)
        else:
            self.number_rel_all -= 1
            print("Empty relation is created and deleted")

    def remove_padding(self, tuples):
        all_zero = np.zeros(tuples.shape[1])
        row_to_delete = []
        for i in range(tuples.shape[0]):
            if (tuples[i] == all_zero).all():
                row_to_delete.append(i)
        return np.delete(tuples, row_to_delete, 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-number_ent', type=int, default=10)
    parser.add_argument('-number_rel', type=int, default=5)
    parser.add_argument('-number_edge', type=int, default=20)
    parser.add_argument('-max_arity', type=int, default=5)
    parser.add_argument('-min_arity', type=int, default=2)
    parser.add_argument('-p_rename', type=float, default=0.30)
    parser.add_argument('-p_projection', type=float, default=0.30)
    parser.add_argument('-p_union', type=float, default=0.10)
    parser.add_argument('-p_product', type=float, default=0.10)
    parser.add_argument('-p_selection', type=float, default=0.30)
    parser.add_argument('-p_setd', type=float, default=0.10)
    parser.add_argument('-output_dir', type=str, default=None, help="A path to the directory where the dataset will be saved and/or loaded from.")
    parser.add_argument('-dataset_name', type=str, default="small")
    args = parser.parse_args()

    synthesizer = Synthesizer(args)
