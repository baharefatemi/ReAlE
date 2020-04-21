import os
import datetime
from random import seed
from random import randint
import argparse
import numpy as np

DEFAULT_SAVE_DIR = './outputs'
seed(3)

class Synthesizer:
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.number_ent = args.number_ent
        self.number_rel = args.number_rel
        self.number_edge = args.number_edge
        self.max_arity = args.max_arity
        self.p_rename = args.p_rename
        self.p_projection = args.p_projection
        self.p_union = args.p_union
        self.p_product = args.p_product
        self.p_selection = args.p_selection
        self.p_setd = args.p_setd

        self.output_dir = self.create_output_dir(args.output_dir)
        self.arities = self.create_arities()
        self.tuples = self.create_init_graph()

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
            arity = randint(1, self.max_arity)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-number_ent', type=int, default=100)
    parser.add_argument('-number_rel', type=int, default=5)
    parser.add_argument('-number_edge', type=int, default=100)
    parser.add_argument('-max_arity', type=int, default=3)
    parser.add_argument('-p_rename', type=float, default=0.10)
    parser.add_argument('-p_projection', type=float, default=0.10)
    parser.add_argument('-p_union', type=float, default=0.10)
    parser.add_argument('-p_product', type=float, default=0.10)
    parser.add_argument('-p_selection', type=float, default=0.10)
    parser.add_argument('-p_setd', type=float, default=0.10)
    parser.add_argument('-output_dir', type=str, default=None, help="A path to the directory where the dataset will be saved and/or loaded from.")
    parser.add_argument('-dataset_name', type=str, default="small")
    args = parser.parse_args()

    synthesizer = Synthesizer(args)
