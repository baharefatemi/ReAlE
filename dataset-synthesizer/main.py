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

        self.p_rename_1 = args.p_rename_1
        self.p_projection_1 = args.p_projection_1
        self.p_union_1 = args.p_union_1
        self.p_product_1 = args.p_product_1
        self.p_selection_1 = args.p_selection_1
        self.p_setd_1 = args.p_setd_1

        self.p_rename_2 = args.p_rename_2
        self.p_projection_2 = args.p_projection_2
        self.p_union_2 = args.p_union_2
        self.p_product_2 = args.p_product_2
        self.p_selection_2 = args.p_selection_2
        self.p_setd_2 = args.p_setd_2


        self.tuples_per_rel = defaultdict(lambda: [])

        
        self.output_dir = self.create_output_dir(args.output_dir)
        self.arities = self.create_arities()
        self.init_tuples = self.create_init_graph()
        self.add_tuples_per_rel(self.init_tuples)

        self.first_layer_operations()
        self.number_rel_1 = self.number_rel_all 

        self.second_layer_operations()

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
        arities = []
        for rel in range(self.number_rel):
            arity = randint(1, self.max_arity//2)
            arities.append(arity)
        return arities

    def create_init_graph(self):
        tuples = np.zeros((self.number_edge,  self.max_arity + 1))
        for edge in range(self.number_edge):
            rel = randint(0, self.number_rel-1)
            entities = []
            arity = self.arities[rel]
            for ar in range(arity):
                entities.append(randint(1, self.number_ent))
            tuples[edge][0] = rel
            tuples[edge][1:arity+1] = entities
        return tuples

    def add_tuples_per_rel(self, tuples):
        for t in tuples:
            # if t[0] not in self.tuples_per_rel:
            #     self.tuples_per_rel[t[0]] = []
            self.tuples_per_rel[t[0]].append(t[1:])

    def first_layer_operations(self):
        self.rename_tuples_1 = np.zeros((0,  self.max_arity + 1))
        self.projection_tuples_1 = np.zeros((0,  self.max_arity + 1))
        self.union_tuples_1 = np.zeros((0,  self.max_arity + 1))
        self.product_tuples_1 = np.zeros((0,  self.max_arity + 1))
        self.selection_tuples_1 = np.zeros((0,  self.max_arity + 1))
        self.setd_tuples_1 = np.zeros((0,  self.max_arity + 1))

        print("Applying first layer operations")

        self.create_rename_graph(1)
        n_rename = self.number_rel_all - self.number_rel
        print("number of rename rels", n_rename)

        self.create_projection_graph(1)
        n_projection = self.number_rel_all - n_rename
        print("number of projection rels", n_projection)
        
        self.create_union_graph(1)
        n_union = self.number_rel_all - n_rename - n_projection
        print("number of union rels", n_union)

        self.create_product_graph(1)
        n_product = self.number_rel_all - n_rename - n_projection - n_union
        print("number of product rels", n_product)

        self.create_selection_graph(1)
        n_selection = self.number_rel_all - n_rename - n_projection - n_union - n_product
        print("number of selection rels", n_selection)

        self.create_setd_graph(1)
        n_setd = self.number_rel_all - n_rename - n_projection - n_union - n_product - n_selection
        print("number of setd rels", n_setd)

        self.add_tuples_per_rel(self.rename_tuples_1)
        self.add_tuples_per_rel(self.projection_tuples_1)
        self.add_tuples_per_rel(self.union_tuples_1)
        self.add_tuples_per_rel(self.product_tuples_1)
        self.add_tuples_per_rel(self.selection_tuples_1)
        self.add_tuples_per_rel(self.setd_tuples_1)


    def second_layer_operations(self):
        self.rename_tuples_2 = np.zeros((0,  self.max_arity + 1))
        self.projection_tuples_2 = np.zeros((0,  self.max_arity + 1))
        self.union_tuples_2 = np.zeros((0,  self.max_arity + 1))
        self.product_tuples_2 = np.zeros((0,  self.max_arity + 1))
        self.selection_tuples_2 = np.zeros((0,  self.max_arity + 1))
        self.setd_tuples_2 = np.zeros((0,  self.max_arity + 1))

        print("Applying second layer operations")

        self.create_rename_graph(2)
        n_rename = self.number_rel_all - self.number_rel_1
        print("number of rename rels", n_rename)

        self.create_projection_graph(2)
        n_projection = self.number_rel_all - self.number_rel_1 - n_rename
        print("number of projection rels", n_projection)
        
        self.create_union_graph(2)
        n_union = self.number_rel_all - self.number_rel_1 - n_rename - n_projection
        print("number of union rels", n_union)

        self.create_product_graph(2)
        n_product = self.number_rel_all - self.number_rel_1 - n_rename - n_projection - n_union
        print("number of product rels", n_product)

        self.create_selection_graph(2)
        n_selection = self.number_rel_all - self.number_rel_1 - n_rename - n_projection - n_union - n_product
        print("number of selection rels", n_selection)

        self.create_setd_graph(2)
        n_setd = self.number_rel_all - self.number_rel_1 - n_rename - n_projection - n_union - n_product - n_selection
        print("number of setd rels", n_setd)

        self.add_tuples_per_rel(self.rename_tuples_2)
        self.add_tuples_per_rel(self.projection_tuples_2)
        self.add_tuples_per_rel(self.union_tuples_2)
        self.add_tuples_per_rel(self.product_tuples_2)
        self.add_tuples_per_rel(self.selection_tuples_2)
        self.add_tuples_per_rel(self.setd_tuples_2)

    def create_rename_graph(self, ind):
        if ind == 1:
            for rel in range(self.number_rel):
                if random.random() < self.p_rename_1:
                    self.rename_tuples_1 = np.concatenate((self.rename_tuples_1, self.rename(rel)))
        elif ind == 2:
            for rel in range(self.number_rel, self.number_rel_1):
                if random.random() < self.p_rename_2:
                    self.rename_tuples_2 = np.concatenate((self.rename_tuples_2, self.rename(rel)))

    def create_projection_graph(self, ind):
        if ind == 1:
            for rel in range(self.number_rel):
                if random.random() < self.p_projection_1:
                    self.projection_tuples_1 = np.concatenate((self.projection_tuples_1, self.projection(rel)))
        elif ind == 2:
            for rel in range(self.number_rel, self.number_rel_1):
                if random.random() < self.p_projection_2:
                    self.projection_tuples_2 = np.concatenate((self.projection_tuples_2, self.projection(rel)))


    def create_union_graph(self, ind):
        if ind == 1:
            for rel1 in range(self.number_rel):
                for rel2 in range(self.number_rel):
                    if rel1 != rel2 and self.arities[rel1] == self.arities[rel2]:
                        if random.random() < self.p_union_1:
                            self.union_tuples_1 = np.concatenate((self.union_tuples_1, self.union(rel1, rel2)))
        elif ind == 2:
            for rel1 in range(self.number_rel, self.number_rel_1):
                for rel2 in range(self.number_rel):
                    if rel1 != rel2 and self.arities[rel1] == self.arities[rel2]:
                        if random.random() < self.p_union_2:
                            self.union_tuples_2 = np.concatenate((self.union_tuples_2, self.union(rel1, rel2)))

    def create_product_graph(self, ind):
        if ind == 1:
            for rel1 in range(self.number_rel):
                for rel2 in range(self.number_rel):
                    if rel1 != rel2 and (self.arities[rel1] + self.arities[rel2]) != self.max_arity:
                        if random.random() < self.p_product_1:
                            self.product_tuples_1 = np.concatenate((self.product_tuples_1, self.product(rel1, rel2)))
        elif ind == 2:
            for rel1 in range(self.number_rel, self.number_rel_1):
                for rel2 in range(self.number_rel):
                    if rel1 != rel2 and (self.arities[rel1] + self.arities[rel2]) != self.max_arity:
                        if random.random() < self.p_product_2:
                            self.product_tuples_2 = np.concatenate((self.product_tuples_1, self.product(rel1, rel2)))


    def create_selection_graph(self, ind):
        if ind == 1:
            for rel in range(self.number_rel):
                if random.random() < self.p_selection_1 and self.arities[rel] > 1:
                    self.selection_tuples_1 = np.concatenate((self.selection_tuples_1, self.selection(rel)))
        elif ind == 2:
            for rel in range(self.number_rel, self.number_rel_1):
                if random.random() < self.p_selection_2 and self.arities[rel] > 1:
                    self.selection_tuples_2 = np.concatenate((self.selection_tuples_2, self.selection(rel)))

    def create_setd_graph(self, ind):
        if ind == 1:
            for rel1 in range(self.number_rel):
                for rel2 in range(self.number_rel):
                    if rel1 != rel2 and self.arities[rel1] == self.arities[rel2]:
                        if random.random() < self.p_setd_1:
                            self.setd_tuples_1 = np.concatenate((self.setd_tuples_1, self.setd(rel1, rel2)))
        elif ind == 2:
            for rel1 in range(self.number_rel, self.number_rel_1):
                for rel2 in range(self.number_rel):
                    if rel1 != rel2 and self.arities[rel1] == self.arities[rel2]:
                        if random.random() < self.p_setd_2:
                            self.setd_tuples_2 = np.concatenate((self.setd_tuples_2, self.setd(rel1, rel2)))


    def create_new_permutation(self, arity):
        all_permutations = list(permutations(range(0, arity)))
        return random.choice(all_permutations)

    def rename(self, rel):
        print("Rename operation for relation {}".format(str(rel)))
        curr_tuples = self.tuples_per_rel[rel]
        rename_tuples = np.zeros((len(curr_tuples),  self.max_arity + 1))
        new_permutation = self.create_new_permutation(self.arities[rel])
        new_rel = self.number_rel_all
        print("New relation id {}".format(str(new_rel)))
        self.arities.append(self.arities[rel])
        self.number_rel_all += 1
        for ind, t in enumerate(curr_tuples):
            entities = []
            for p in new_permutation:
                entities.append(t[p])
            rename_tuples[ind][0] = new_rel
            rename_tuples[ind][1:self.arities[rel]+1] =  entities
        return rename_tuples

    def projection(self, rel):
        print("Projection operation for relation {}".format(str(rel)))
        curr_tuples = self.tuples_per_rel[rel]
        projection_tuples = np.zeros((len(curr_tuples),  self.max_arity + 1))
        new_rel_arity = randint(1, self.arities[rel])
        new_rel = self.number_rel_all
        print("New relation id {}".format(str(new_rel)))
        self.arities.append(new_rel_arity)
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

        return projection_tuples

    def union(self, rel1, rel2):
        print("Union operation for relation {} and {}".format(str(rel1), str(rel2)))
        curr_tuples1 = self.tuples_per_rel[rel1]
        curr_tuples2 = self.tuples_per_rel[rel2]
        union_tuples = np.zeros((len(curr_tuples1) + len(curr_tuples2),  self.max_arity + 1))
        new_rel = self.number_rel_all
        self.arities.append(self.arities[rel1])
        print("New relation id {}".format(str(new_rel)))
        self.number_rel_all += 1

        for ind, t in enumerate(curr_tuples1):
            union_tuples[ind][1:] = t
            union_tuples[ind][0] = new_rel
        for ind, t in enumerate(curr_tuples2):
            union_tuples[len(curr_tuples1) + ind][1:] = t
            union_tuples[len(curr_tuples1) + ind][0] = new_rel
        return union_tuples

    def product(self, rel1, rel2):
        print("Product operation for relation {} and {}".format(str(rel1), str(rel2)))
        curr_tuples1 = self.tuples_per_rel[rel1]
        curr_tuples2 = self.tuples_per_rel[rel2]
        product_tuples = np.zeros((len(curr_tuples1) * len(curr_tuples2),  self.max_arity + 1))
        new_rel = self.number_rel_all
        self.arities.append(self.arities[rel1] + self.arities[rel2])
        print("New relation id {}".format(str(new_rel)))
        self.number_rel_all += 1
        for ind1, t1 in enumerate(curr_tuples1):
            for ind2, t2 in enumerate(curr_tuples2):
                product_tuples[(ind1 * len(curr_tuples2))+ ind2][0] = new_rel
                product_tuples[(ind1 * len(curr_tuples2))+ ind2][1:self.arities[rel1]+1] = t1[0:self.arities[rel1]]
                product_tuples[(ind1 * len(curr_tuples2))+ ind2][self.arities[rel1]+1:self.arities[rel1] + self.arities[rel2]+1] = t2[0:self.arities[rel2]]

        return product_tuples

    def selection(self, rel):
        print("Selection operation for relation {}".format(str(rel)))
        curr_tuples = self.tuples_per_rel[rel]
        selection_tuples = np.zeros((len(curr_tuples),  self.max_arity + 1))
        new_rel = self.number_rel_all
        print("New relation id {}".format(str(new_rel)))
        self.arities.append(self.arities[rel])
        self.number_rel_all += 1

        selection_type = random.random()
        if selection_type < 0.50:
            which_pos_1 = 0
            which_pos_2 = 0
            while which_pos_1 == which_pos_2:
                which_pos_1 = randint(0, self.arities[rel]-1)
                which_pos_2 = randint(0, self.arities[rel]-1)
            for ind, t in enumerate(curr_tuples):
                if t[which_pos_1] == t[which_pos_2]:
                    selection_tuples[ind][0] = new_rel
                    selection_tuples[ind][1:] = t
        else:
            which_pos_1 = randint(0, self.arities[rel])
            which_val = randint(0, self.number_ent - 1)
            for ind, t in enumerate(curr_tuples):
                if t[which_pos_1] == which_val:
                    selection_tuples[ind][0] = new_rel
                    selection_tuples[ind][1:] = t
        # remove one selected column
        selection_tuples = self.remove_one_column(selection_tuples, which_pos_1)
        return self.remove_padding(selection_tuples)

    def remove_one_column(self, tuples, col):
        tuples[:, col] = np.zeros(tuples.shape[0])
        tuples[:, col:] = np.roll(tuples[:,col:], -1, axis=1)
        return tuples

    def setd(self, rel1, rel2):
        print("Setd operation for relation {} and {}".format(str(rel1), str(rel2)))
        curr_tuples1 = self.tuples_per_rel[rel1]
        curr_tuples2 = self.tuples_per_rel[rel2]
        setd_tuples = np.zeros((len(curr_tuples1),  self.max_arity + 1))
        new_rel = self.number_rel_all
        print("New relation id {}".format(str(new_rel)))
        self.arities.append(self.arities[rel1])
        self.number_rel_all += 1

        for ind, t in enumerate(curr_tuples1):
            # if not np.any(curr_tuples2 == np.array(t)):
            curr_tuples2 = np.array(curr_tuples2)
            if not(tuple(np.array(t)) in {tuple(v): True for v in curr_tuples2}):
                setd_tuples[ind][1:] = t
                setd_tuples[ind][0] = new_rel
 
        return self.remove_padding(setd_tuples)


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
    parser.add_argument('-number_rel', type=int, default=3)
    parser.add_argument('-number_edge', type=int, default=20)
    parser.add_argument('-max_arity', type=int, default=5)
    parser.add_argument('-p_rename_1', type=float, default=0.30)
    parser.add_argument('-p_projection_1', type=float, default=0.30)
    parser.add_argument('-p_union_1', type=float, default=0.10)
    parser.add_argument('-p_product_1', type=float, default=0.10)
    parser.add_argument('-p_selection_1', type=float, default=0.30)
    parser.add_argument('-p_setd_1', type=float, default=0.10)
    parser.add_argument('-p_rename_2', type=float, default=0.30)
    parser.add_argument('-p_projection_2', type=float, default=0.30)
    parser.add_argument('-p_union_2', type=float, default=0.10)
    parser.add_argument('-p_product_2', type=float, default=0.10)
    parser.add_argument('-p_selection_2', type=float, default=0.30)
    parser.add_argument('-p_setd_2', type=float, default=0.10)
    parser.add_argument('-output_dir', type=str, default=None, help="A path to the directory where the dataset will be saved and/or loaded from.")
    parser.add_argument('-dataset_name', type=str, default="small")
    args = parser.parse_args()

    synthesizer = Synthesizer(args)
