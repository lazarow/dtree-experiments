from random import choice, choices, randint, randrange, random, uniform
from typing import Tuple, List
from statistics import mode

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

maxdepth = 100 # Max tree depth in decision tree with GA
EPSILON = 0.001 # convergence tolerance

# Fitness fucntion parameters
errorfactor = 1 # the factor that controls the relative importance of model accuracy
heightfactor = 0.025 #regularisation term, it penalizes the size of an individual in terms of depth of the tree

class Node:
    def __init__(self):
        self.parent: Node = None
        self.attribute = None
        self.threshold = None
        self.child_yes: Node = None
        self.child_no: Node = None
        self.label = None

    def copy(self):
        new = Node()
        new.attribute = self.attribute
        new.threshold = self.threshold
        new.label = self.label

        if new.label is None:
            new.child_yes = self.child_yes.copy()
            new.child_yes.parent = new
            new.child_no = self.child_no.copy()
            new.child_no.parent = new

        return new

    def height(self) -> int:
        if self.label is not None:
            return 1
        else:
            return 1 + max(self.child_yes.height(), self.child_no.height())

    def subnodes_count(self) -> int:
        if self.label:
            return 1
        else:
            return self.child_yes.subnodes_count() + self.child_no.subnodes_count()


def get_nth_subnode(root: Node, n: int) -> Node:
    tmp: List[Node] = [root]

    for _ in range(n):
        cur = tmp.pop()
        if cur.label is None:
            tmp.append(cur.child_no)
            tmp.append(cur.child_yes)

    return cur


def generate_subtree(p_split: float, attributes: int,
                     ranges: List[Tuple[float, float]], labels: np.ndarray,
                     depth: int = 1) -> Node:
    MAX_DEPTH = maxdepth
    node = Node()

    if random() < p_split and depth < MAX_DEPTH:
        node.attribute = randrange(attributes)
        node.threshold = uniform(*ranges[node.attribute])
        node.child_yes = generate_subtree(p_split, attributes, ranges, labels,
                                          depth + 1)
        node.child_yes.parent = node
        node.child_no = generate_subtree(p_split, attributes, ranges, labels,
                                         depth + 1)
        node.child_no.parent = node
    else:
        node.label = choice(labels)

    return node

class Tree:
    def __init__(self, root: Node, value: float):
        self.root = root
        self.value = value

    def copy(self):
        return Tree(self.root.copy(), self.value)

class EDT(BaseEstimator, ClassifierMixin):
    """Evolutionary Decision Tree"""
    
    def __init__(
        self,
        N: int = 500,
        Tournament_number: int = 25,
        p_split: float = 0.5,
        target_height: int = 15,
        tournament_k: int = 9,
        mutation_prob: float = 0.05,
        M: int = 1000,
        stall_iter: int = 100
    ):
        self.N = N
        self.Tournament_number = Tournament_number + (Tournament_number % 2)  # make it even
        self.p_split = p_split
        self.target_height = target_height
        self.tournament_k = tournament_k
        self.mutation_prob = mutation_prob
        self.M = M
        self.stall_iter = stall_iter
        self.root: Node = None

    def get_params(self, deep = True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"N": self.N, 
                "Tournament_number": self.Tournament_number, 
                "p_split": self.p_split, 
                "target_height": self.target_height,
                "tournament_k": self.tournament_k,
                "mutation_prob": self.mutation_prob,
                "M": self.M,
                "stall_iter": self.stall_iter}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def eval_try(self, x: np.ndarray) -> float:
        """Returns error of prediction of x given true values y."""
        return self.eval_from_node_try(self.root, x)
            
    def eval(self, x: np.ndarray, y: np.ndarray) -> float:
        """Returns error of prediction of x given true values y."""
        return self.eval_from_node(self.root, x, y)

    def predict(self, x: np.ndarray):
        """Predicts output for input x."""
        if self.root is None:
            raise Exception('Model not trained!')
        return self.predict_from_node(self.root, x)

    def fit(self, x: np.ndarray, y: np.ndarray, verbose: bool = False) -> None:
        """Finds decision tree that tries to predict y given x."""
        attributes = len(x[0])
        ranges = []
        for i in range(attributes):
            vals = [tmp[i] for tmp in x]
            ranges.append((min(vals), max(vals)))
        labels = np.unique(y)

        P = []
        for _ in range(self.N):
            root = generate_subtree(self.p_split, attributes, ranges, labels)
            value = self.ga_fun(root, x, y)
            P.append(Tree(root, value))

        stop = False
        iter = 1
        stall_iter = 0
        current_best = P[0]
        best_val = current_best.value

        while not stop:
            try:
                R = self.select(P)
                C = self.crossover(R, x, y)
                O = self.mutation(C, x, y, attributes, ranges, labels)
                P = self.replace(P, O)

                P.sort(key=lambda tree: tree.value)
                current_best = P[0]

                if verbose:
                    self.diagnostics(iter, P)

                if abs(current_best.value - best_val) < EPSILON:
                    stall_iter += 1
                else:
                    stall_iter = 0
                    best_val = current_best.value

                if iter >= self.M or stall_iter >= self.stall_iter:
                    stop = True
                iter += 1
            except KeyboardInterrupt:
                print('User interrupted!')
                stop = True
        depths = {tmp.root.height() for tmp in P}
        self.root = current_best.root

    def diagnostics(self, iter: int, P: List[Tree]) -> None:
        vals = [tmp.value for tmp in P]
        mean = sum(vals) / len(P)
        best = min(vals)
        depths = {tmp.root.height() for tmp in P}

        print(f"[Iteration {iter:02d}] "
              f"Best fitness value: {best:.5f}, "
              f"Mean fitness: {mean:.3f}, "
              f"Depths: {sorted(depths)}")

    def verify_values(self, trees: List[Tree], x: np.ndarray, y: np.ndarray):
        result = all(
            abs(tree.value - self.ga_fun(tree.root, x, y)) < 0.01 for tree in
            trees)
        print(f'All correct: {result}')

# selection function
    def select(self, P: List[Tree]) -> List[Tree]:
        R = []

        for _ in range(self.Tournament_number):
            rank = choices(P, k=self.tournament_k)
            rank.sort(key=lambda tree: tree.value)
            R.append(rank[0].copy())
        return R

#crossover function
    def crossover(self, R: List[Tree], x: np.ndarray, y: np.ndarray) -> List[Tree]:
        R = [tree.copy() for tree in R]
        pairs = [(R[2 * i], R[2 * i + 1]) for i in range(int(len(R) / 2))]
        C = []

        for a, b in pairs:
            first = get_nth_subnode(a.root,
                                    randint(1, a.root.subnodes_count()))
            second = get_nth_subnode(b.root,
                                     randint(1, b.root.subnodes_count()))

            self.swap(a, first, b, second)

            C.append(Tree(a.root, self.ga_fun(a.root, x, y)))
            C.append(Tree(b.root, self.ga_fun(b.root, x, y)))

        return C

    def swap(self, a_tree, a_node, b_tree, b_node):
        if a_node.parent is not None and b_node.parent is not None:
            a_node.parent, b_node.parent = b_node.parent, a_node.parent

            if b_node is a_node.parent.child_yes:
                a_node.parent.child_yes = a_node
            else:
                a_node.parent.child_no = a_node

            if a_node is b_node.parent.child_yes:
                b_node.parent.child_yes = b_node
            else:
                b_node.parent.child_no = b_node
        if a_node.parent is None and b_node.parent is not None:
            a_node.parent, b_node.parent = b_node.parent, a_node.parent
            # b_node.parent == None
            if b_node is a_node.parent.child_yes:
                a_node.parent.child_yes = a_node
            else:
                a_node.parent.child_yes = a_node

            a_tree.root = b_node
        if a_node.parent is not None and b_node.parent is None:
            a_node.parent, b_node.parent = b_node.parent, a_node.parent
            # a_node.parent == None
            if a_node is b_node.parent.child_yes:
                b_node.parent.child_yes = b_node
            else:
                b_node.parent.child_no = b_node

            b_tree.root = a_node
        if a_node.parent is None and b_node.parent is None:
            a_tree.root, b_tree.root = b_node, a_node

#mutation function
    def mutation(self, C: List[Tree], x: np.ndarray, y: np.ndarray,
                 attributes: int, ranges: List[Tuple[float, float]],
                 labels: list) -> List[Tree]:
        C = [tree.copy() for tree in C]
        O = []

        for tree in C:
            tmp: List[Node] = [tree.root]
            while len(tmp) > 0:
                cur = tmp.pop()

                if random() < self.mutation_prob:
                    if cur.label is None:
                        cur.attribute = randrange(attributes)
                        cur.threshold = uniform(*ranges[cur.attribute])
                    else:
                        cur.label = choice(labels)

                if cur.label is None:
                    tmp.append(cur.child_no)
                    tmp.append(cur.child_yes)

            O.append(Tree(tree.root, self.ga_fun(tree.root, x, y)))

        return O

# recombination function for mutation
    def replace(self, P: List[Tree], O: List[Tree]) -> List[Tree]:
        union = P + O
        union.sort(key=lambda tree: tree.value)
        return union[:self.N]

    def predict_from_node(self, root: Node, x: np.ndarray):
        node = root
        while node.label is None:
            if x[node.attribute] > node.threshold:
                node = node.child_yes
            else:
                node = node.child_no

        return node.label

    
    def eval_from_node(self, root: Node, x: np.ndarray, y: np.ndarray) -> float:
        assert len(x) == len(y)
        preds = [self.predict_from_node(root, sample) for sample in x]
        errors = [pred != goal for pred, goal in zip(preds, y)]
        return sum(errors) / len(errors)
    
    def eval_from_node_try(self, root: Node, x: np.ndarray) -> float:
        preds = [self.predict_from_node(root, sample) for sample in x]
        return (preds)

# fitness function -> evaluating individual fitness score
    def ga_fun(self, root: Node, x: np.ndarray, y: np.ndarray) -> float:
        error_factor = errorfactor * self.eval_from_node(root, x, y)
        height_factor = heightfactor * root.height() / self.target_height
        return error_factor + height_factor


print('Done.')