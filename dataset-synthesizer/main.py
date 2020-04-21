
class Synthesizer:
    def __init__(self, args):
        self.number_ent = args.number_ent
        self.number_rel = args.number_rel
        self.max_arity = args.max_arity
        self.p_rename = args.p_rename
        self.p_projection = args.p_projection
        self.p_union = args.p_union
        self.p_product = args.p_product
        self.p_selection = args.p_selection
        self.p_setd = args.p_setd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-number_ent', type=int, default=100)
    parser.add_argument('-number_rel', type=int, default=5)
    parser.add_argument('-max_arity', type=int, default=3)
    parser.add_argument('-p_rename', type=float, default=0.10)
    parser.add_argument('-p_projection', type=float, default=0.10)
    parser.add_argument('-p_union', type=float, default=0.10)
    parser.add_argument('p_product', type=float, default=0.10)
    parser.add_argument('p_selection', type=float, default=0.10)
    parser.add_argument('p_setd', type=float, default=0.10)



    synthesizer = synthesizer(args)
