import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Optional, Union

# get currentdir
import os 
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0,parentdir)
from simtransformer.module_base import ConfigBase
from simtransformer.utils import EasyDict, clever_save, clever_load, shuffle_with_indices

unk_token = '<unk>'
pad_token = '<pad>'
eos_token = '<eos>'

class Tokenizer():
    def __init__(self, vocab, unk_token=unk_token, pad_token=pad_token, eos_token=eos_token):
        self.vocab = vocab + [unk_token, pad_token, eos_token]

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.eos_token = eos_token

        self.tok2idx = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.idx2tok = {idx: tok for idx, tok in enumerate(self.vocab)}

        self.unk_token_idx = self.tok2idx[self.unk_token]

    def encode_tokens(self, list_tokens):
        """Given a list of strings, each representing a token, return a list of indices corresponding to the tokens in the vocabulary."""
        return [self.tok2idx.get(tok, self.tok2idx[self.unk_token]) for tok in list_tokens]

    def decode_tokens(self, list_indices):
        """Given a list of indices, return a list of string tokens corresponding to the indices in the vocabulary."""
        return [self.idx2tok.get(idx, self.unk_token) for idx in list_indices]

    def encode_string(self, string, sep=' '):
        return self.encode_tokens(string.split(sep))


class ADD:
    def __init__(self, mod_val) -> None:
        self.mod_val = int(mod_val)
    def __call__(self, x, y):
        return int((x + y) % self.mod_val)
    @property
    def __name__(self):
        return "ADD"

class MUL:
    def __init__(self, mod_val) -> None:
        self.mod_val = int(mod_val)
    def __call__(self, x, y):
        return int((x * y) % self.mod_val)
    @property
    def __name__(self):
        return "MUL"
    
class EQUL:
    def __init__(self, mod_val) -> None:
        self.mod_val = int(mod_val)
    def __call__(self, x, y):
        return int(y % self.mod_val)
    @property
    def __name__(self):
        return "EQUL"

class DAGWeightedNode:
    def __init__(self, name, weight=None, mod_val=19):
        """
        Initialize a DAG node with a name and an optional weight.
        
        Parameters:
        - name: the name of the node
        - weight: the weight of the node (default is None)
        """
        self.mod_val = mod_val
        self.name = name
        self.weight = weight if weight is not None else random.randint(0, mod_val - 1)
        self.fan_in = []  # Stores the fan-in nodes and the applied functions
        self.depth = 0  # Depth of the node, default is 0
        self.oper_depth = 0  # Total number of operations needed to compute the node's value

    def add_fan_in(self, parent_node, func):
        """
        Add a fan-in node with the function used to combine the values. 
        
        Parameters:
        - parent_node: the DAGWeightedNode instance of the parent node
        - func: the function used to combine the parent node's value with the current value
        """
        self.fan_in.append((parent_node, func))

    def compute_weight(self):
        """
        Compute the value of the node based on its fan-in nodes and functions.
        
        Returns:
        - The computed value of the node
        """
        if len(self.fan_in) == 0:
            pass
        elif len(self.fan_in) > 0:
            for parent_node, func in self.fan_in:
                self.weight = func.__call__(self.weight, parent_node.weight)

    def compute_depth(self):
        """
        Compute the depth of the node based on its fan-in nodes.
        """
        self.depth = 0
        if len(self.fan_in) == 0:
            pass
        elif len(self.fan_in) > 0:
            for parent_node, func in self.fan_in:
                self.depth = max(self.depth, parent_node.depth + 1)
    
    def compute_oper_depth(self):
        """
        Compute the total number of operations needed to compute the node's value.
        """
        self.oper_depth = 1
        if len(self.fan_in) == 0:
            pass # No fan-in nodes means only need one operation to assign the weight
        elif len(self.fan_in) > 0:
            self.oper_depth = len(self.fan_in) # value assignment EQUL is also counted as an operation
            for parent_node, func in self.fan_in:
                self.oper_depth += parent_node.oper_depth

    def print_algorithmic_expression(self):
        """
        Print an algorithmic expression that represents how the fan-in is calculated.
        """
        if not self.fan_in:
            print(f"{self.depth:<5}{self.name:<5} = {self.weight:<5} / No fan-in")
        else:
            value = 0
            expression = ""
            for parent_node, func in self.fan_in:
                operation = func.__name__ if hasattr(func, '__name__') else 'func'
                expression += f" {operation} ({parent_node.name})"
                value = func.__call__(value, parent_node.weight)
            expression = f"{self.depth:<5}{self.name:<5} = {self.weight:<5} / {value: < 5} <-" + expression
            print(expression)

    def to_math_expression(self, inverse=False):
        """
        Convert the node's information into a mathematical expression.
        
        Returns:
        - A string representing the mathematical expression of the node.
        """
        if not self.fan_in:
            return f"{self.weight} = {self.name}"
        else:
            expression = ''
            for i, (parent_node, func) in enumerate(self.fan_in):
                operation = func.__name__ if hasattr(func, '__name__') else 'func'
                if i > 0:
                    expression += f" {operation} "
                expression += f"{parent_node.name}"
            expression += f" = {self.name}"
            return expression

    @classmethod
    def from_math_expression(cls, expression, node_dict, mod_val=19):
        """
        Create or update a node from a given mathematical expression. The 
        
        Parameters:
        - expression: A string representing the mathematical expression of the node.
        - node_dict: A dictionary of existing nodes to reference.
        
        Returns:
        - The updated DAGWeightedNode instance.
        """
        parts = expression.split("=")
        node_name = parts[-1].strip()
        rhs = parts[0].strip()
        terms = rhs.split()
        node = cls(node_name)
        
        i = 0
        while i < len(terms):
            if terms[i] in ['ADD', 'MUL', 'EQUL']:
                operation = terms[i]
                i += 1
                parent_name = terms[i]
                func = {'ADD': ADD, 'MUL': MUL, 'EQUL': EQUL}[operation](mod_val)
                node.add_fan_in(node_dict[parent_name], func)
            else:
                parent_name = terms[i]
                if parent_name.isdigit():
                    node.weight = int(parent_name)
                else:
                    node.add_fan_in(node_dict[parent_name], EQUL(mod_val))
            i += 1
        node.compute_weight()
        return node
    
class ArithmeticDAG:
    def __init__(self, 
                 vocab: List, 
                 min_fan_in_deg: int = 1, 
                 max_fan_in_deg: int = 3, 
                 num_leaf_nodes: int = 5,
                 max_depth: int = 32,
                 func_vocab: Optional[List] = [],
                 mod_val: Optional[int] = 19,
                 shuffle_predecessors: Optional[bool] = False,
                 verbose: Optional[bool] = False,
                 vocab_obj: Optional[DAGWeightedNode] = None,
                 class_type: str = 'abs', 
                    **kwargs
                 ):
        """
        Initialize a Directed Acyclic Graph (DAG) in dictionary order, node by node.
        
        Parameters:
        - vocab: list of nodes (e.g., ['a', 'b', 'c', 'd'])
        - min_fan_in_deg: minimum number of incoming edges per node (must be at least 0)
        - max_fan_in_deg: maximum number of incoming edges per node (must be at least 0)
        - func_vocab: list of functions to be used for combining fan-in nodes
        """
        self.mod_val = mod_val
        self.num_leaf_nodes = num_leaf_nodes
        self.min_fan_in_deg = min_fan_in_deg
        self.max_fan_in_deg = max_fan_in_deg
        self.max_depth = max_depth

        # if seed is not None:
        #     random.seed(seed)
        
        self.func_vocab = func_vocab if func_vocab else [ADD, MUL]  # Default to addition and multiplication

        if vocab_obj:
            self.node_info = {name: node for name, node in (vocab_obj.items() or [])}
        else: 
            self.node_info = {}

        for node in vocab:
            if node not in self.node_info:
                self.node_info[node] = DAGWeightedNode(node, mod_val=self.mod_val)
                
        if shuffle_predecessors:
            # shuffle the node_info dictionary to randomize the order of the nodes
            shuffled_items = list(self.node_info.items())
            random.shuffle(shuffled_items)
            self.node_info = dict(shuffled_items)
        
        self.vocab = [node.name for node in self.node_info.values()] # already in order

        self.graph = self._generate_random_dag()
        self._init_fan_in_method()
        self.sync_node_states()

        self.class_type = class_type

        if verbose:
            self.draw()
            self.print_nodes()


    def _generate_random_dag(self):
        """
        Generate a random Directed Acyclic Graph (DAG) in dictionary order.
        
        Returns:
        - G: A directed acyclic graph (DAG)
        """
        # Step 1: Initialize the directed graph
        G = nx.DiGraph()
        
        # Step 2: Add nodes from the abstract vocabulary in dictionary order
        for idx, node in enumerate(self.vocab):
            G.add_node(node)
            
            # Step 3: Add edges from previous nodes to the current node
            if idx > self.num_leaf_nodes - 1:
                # Previous nodes available for connecting
                possible_parents = self.vocab[:idx]
                possible_parents = [parent for parent in possible_parents if self.node_info[parent].depth < self.max_depth]
                if len(possible_parents) == 0:
                    raise ValueError("No available parent nodes to connect to, increase the maximum depth")

                # Determine the number of incoming edges (fan-in degree)
                min_deg = min(self.min_fan_in_deg, len(possible_parents))  # Ensure it does not exceed the number of available nodes
                max_deg = min(self.max_fan_in_deg, len(possible_parents))  # Ensure it does not exceed the number of available nodes
                
                # Randomly select the number of parents within bounds
                num_parents = random.randint(min_deg, max_deg)
                
                # Randomly select parent nodes and add edges
                parents = random.sample(possible_parents, num_parents)
                for parent in parents:
                    G.add_edge(parent, node)
                    self.node_info[node].depth = max(self.node_info[node].depth, self.node_info[parent].depth + 1)

        return G

    def update_depths_from_root(self):
        """
        Update the depth of each non-leaf node starting from the root node.
        """
        for node in nx.topological_sort(self.graph):
            if not list(self.graph.successors(node)):
                continue
            self.node_info[node].depth = max([self.node_info[parent].depth for parent in self.graph.predecessors(node)]) + 1

    def _init_fan_in_method(self):
        """
        Init the fan-in method for each node and assign random operations to combine them.
        """
        for node in self.graph.nodes:
            fan_in_nodes = list(self.graph.predecessors(node))
            if len(fan_in_nodes) == 1:
                # just set node_info[node].fan_in to be the parent node with no operation
                self.node_info[node].add_fan_in(self.node_info[fan_in_nodes[0]], EQUL(self.mod_val))
            if len(fan_in_nodes) > 1:
                # Assign a random order to fan-in nodes
                random.shuffle(fan_in_nodes)
                
                self.node_info[node].add_fan_in(self.node_info[fan_in_nodes[0]], EQUL(self.mod_val))
                # Combine fan-in nodes using functions from func_vocab
                for j in range(1, len(fan_in_nodes)):
                    func = random.choice(self.func_vocab)
                    self.node_info[node].add_fan_in(self.node_info[fan_in_nodes[j]], func(self.mod_val))

    def sync_node_states(self):
        """
        Iteratively apply each node's compute_weight() method to sync all the values in the graph.
        """
        for node in nx.topological_sort(self.graph):
            self.node_info[node].compute_weight()
            self.node_info[node].compute_depth()
            self.node_info[node].compute_oper_depth()
            
    def draw(self):
        """
        Draw the DAG using matplotlib, with node weights displayed.
        """
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.graph)
        node_labels = {node: f"{node}\n(weight={self.node_info[node].weight:.2f}, depth={self.node_info[node].depth})" for node in self.graph.nodes}
        nx.draw(self.graph, pos, with_labels=True, labels=node_labels, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
        plt.title("Directed Acyclic Graph (DAG) with Node Weights and Depth")
        plt.show()
    
    def print_nodes(self):
        """
        Print the algorithmic expressions for all nodes in the graph.
        """
        for node in self.graph.nodes:
            self.node_info[node].print_algorithmic_expression()

    @property
    def leaf_nodes_string(self):
        """
        Return the leaf nodes of the graph.
        
        Returns:
        - A list of leaf nodes
        """
        return self.vocab[:self.num_leaf_nodes]
    
    @property
    def leaf_nodes_dict(self):
        """
        Return the leaf nodes of the graph as pointers to the DAGWeightedNode instances.
        
        Returns:
        - A list of pointers to the leaf nodes
        """
        return {node: self.node_info[node] for node in self.leaf_nodes_string}
    
    def set_leaf_nodes_value(self, values):
        """
        Set the values of the leaf nodes.
        
        Parameters:
        - values: A list of values for the leaf nodes
        """
        assert len(values) == self.num_leaf_nodes, "Number of values must match the number of leaf nodes"
        for i, node_tuple in enumerate(self.leaf_nodes_dict.items()):
            node_name, node = node_tuple
            node.weight = values[i] % self.mod_val

    def generate_data(self, 
                          splitter: str = ' , ', 
                          shuffle: bool = False, 
                          to_string: bool = True, ) -> str:
        """
        Generate a sentence by iteratively calling `to_math_expression` method of all nodes.

        Parameters:
        - splitter: The string used to split each node's math expression.
        - class_type: The type of class for generating the expression.

        Returns:
        - A string representing the concatenated mathematical expressions of all nodes.
        """
        sorted_node = list(nx.topological_sort(self.graph))
        expressions = [self.node_info[node].to_math_expression() for node in sorted_node]
        depths = [self.node_info[node].depth for node in sorted_node]
        values = [self.node_info[node].weight for node in sorted_node]
        opers = [self.node_info[node].oper_depth for node in sorted_node]

        original_indices = list(range(len(expressions)))
        if shuffle:
            expressions, original_indices = shuffle_with_indices(expressions, original_indices)
        if to_string:
            return splitter.join(expressions), original_indices, depths, values
        else: 
            return expressions, original_indices, depths, values, opers
            

    def expand_graph_from_sentence(self, sentence: str, original_indices: List[int], splitter: str = ', ', aug_node_info: dict = {}) -> None:
        """
        Reconstruct the graph from a given sentence and original indices.

        Parameters:
        - sentence: The concatenated mathematical expressions of all nodes.
        - original_indices: The original order of nodes before shuffling.
        - splitter: The string used to split each node's math expression.
        """
        expressions = sentence.split(splitter) if isinstance(sentence, str) else sentence
        sorted_expressions = [expressions[i] for i in original_indices]

        # Clear existing nodes and rebuild from expressions
        for expr in sorted_expressions:
            node = DAGWeightedNode.from_math_expression(expr, {**self.node_info, **aug_node_info}, self.mod_val)
            self.node_info[node.name] = node

        self.sync_node_states()

    @classmethod
    def from_sentence(cls, 
                      sentence: str, 
                      original_indices: List[int], 
                      splitter: str = ', ', 
                      aug_node_info: dict = {},
                      mod_val: int = 19,
                      **kwargs) -> 'ArithmeticDAG':
        """
        Reconstruct the graph from a given sentence and original indices.

        Parameters:
        - sentence: The concatenated mathematical expressions of all nodes.
        - original_indices: The original order of nodes before shuffling.
        - splitter: The string used to split each node's math expression.
        - aug_node_info: Additional node information to be used during reconstruction.
        - mod_val: The modulus value for the operations.

        Returns:
        - An instance of ArithmeticDAG.
        """
        expressions = sentence.split(splitter) if isinstance(sentence, str) else sentence
        sorted_expressions = [expressions[i] for i in original_indices]

        # Initialize an empty ArithmeticDAG instance
        node_info = {}

        # Clear existing nodes and rebuild from expressions
        for expr in sorted_expressions:
            node = DAGWeightedNode.from_math_expression(expr, {**node_info, **aug_node_info}, mod_val)
            node_info[node.name] = node

        instance = cls(vocab=[], mod_val=mod_val, vocab_obj=node_info, verbose=False, **kwargs)
        instance.sync_node_states()
        # check if kwargs has verbose key
        if 'verbose' in kwargs:
            if kwargs['verbose']:
                instance.print_nodes()
                instance.draw()
        return instance

def generate_abs_dag(data_config: EasyDict):

    # Preparing the vocab for the DAG
    num_nodes = data_config.num_nodes
    vocab = [f'h_{i + 1}' for i in range(num_nodes)]

    if data_config.func_vocab:
        func_vocab = [globals()[func_name] for func_name in data_config.func_vocab]
    else:
        func_vocab = None
    # Initializing the first ArithmeticDAG object
    abs_dag = ArithmeticDAG(
        vocab=vocab,
        min_fan_in_deg=data_config.min_fan_in_deg,
        max_fan_in_deg=data_config.max_fan_in_deg,
        num_leaf_nodes=data_config.num_leaf_nodes,
        max_depth=data_config.max_depth,
        func_vocab=func_vocab,
        mod_val=data_config.mod_val,
        shuffle_predecessors=False,
        verbose=data_config.verbose
    )
    return abs_dag

def generate_ins_dag(abs_dag, data_config: EasyDict):
    # Preparing the vocab for the ins_dag
    num_nodes = data_config.num_nodes
    vocab = [f'v_{i + 1}' for i in range(num_nodes)]

    if data_config.func_vocab:
        func_vocab = [globals()[func_name] for func_name in data_config.func_vocab]
    else:
        func_vocab = None
    # Initializing the ins_dag with the leaf nodes of abs_dag as vocab_obj
    ins_dag = ArithmeticDAG(
        vocab=vocab,
        min_fan_in_deg=data_config.min_fan_in_deg,
        max_fan_in_deg=data_config.max_fan_in_deg,
        num_leaf_nodes=data_config.num_leaf_nodes,
        func_vocab=func_vocab,
        mod_val=data_config.mod_val,
        shuffle_predecessors=True,
        verbose=data_config.verbose,
        vocab_obj=abs_dag.leaf_nodes_dict
    )
    return ins_dag

def generate_simple_dataset(config_path: str):
    currentdir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.path.join(currentdir, config_path)
    config = ConfigBase(config_dir)
    
    data_config = config.data_config
    dag_config = data_config.dag_config
    dag = generate_abs_dag(dag_config)
    
    num_samples = data_config.num_samples
    dataset = []
    
    for _ in range(num_samples):
        sentence, original_indices, depths, values, opers = dag.generate_data(shuffle=False, to_string=False)
        data_piece = {
            'eqs': sentence,
            'depths': depths,
            'values': values,
            'opers': opers,
        }
        dataset.append(data_piece)
        
    data_path = os.path.join(currentdir, dag_config.data_dir, dag_config.data_file_name)
    clever_save(dataset, data_path)
    print(f"Dataset saved to {data_path}")
    
    vocab_path = os.path.join(currentdir, dag_config.data_dir, dag_config.vocab_file_name)
    
    vocab = dag.vocab + [f'{i}' for i in range(dag_config.mod_val)] + ['<eos>', '<pad>', ',', '=', 'ADD', 'MUL']
    clever_save(vocab, vocab_path)
    print(f"Vocab saved to {vocab_path}")



# def generate_hirarchy_dataset(config_path: str):
#     currentdir = os.path.dirname(os.path.realpath(__file__))
#     config_dir = os.path.join(currentdir, config_path)
#     config = ConfigBase(config_dir)

#     data_config = config.data_config
#     abs_dag = generate_abs_dag(data_config.abs_dag_config)

#     # Fixed abs_dag
#     num_samples = data_config.num_samples
#     dataset = []

#     for _ in range(num_samples):
#         ins_dag = generate_ins_dag(abs_dag, data_config.ins_dag_config)

#         # Generate sentence and original indices
#         sentence, original_indices = ins_dag.generate_data(shuffle=True)
#         data_piece = {
#             'sentence': sentence,
#             'original_indices': original_indices
#         }

#         # Sync node states for both DAGs
#         ins_dag.sync_node_values()
#         abs_dag.sync_node_values()

#         # Randomly pick one node from abs_dag
#         random_node = random.choice(list(abs_dag.node_info.values()))
#         data_piece['question'] = random_node.name
#         data_piece['answer'] = random_node.weight

#         # Append data piece to dataset
#         dataset.append(data_piece)

#     # Save dataset to specified path
#     data_path = data_config.data_path
#     with open(data_path, 'w') as f:
#         json.dump(dataset, f, indent=4)

#     print(f"Dataset saved to {data_path}")
    
if __name__ == "__main__":
    
    currentdir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.path.join(currentdir, 'configurations')
    generate_simple_dataset(config_dir)

    config = ConfigBase(config_dir)
    # data_config = config.data_config
    # abs_dag = generate_abs_dag(data_config.abs_dag_config)
    # ins_dag = generate_ins_dag(abs_dag, data_config.ins_dag_config)

    # # set the leaf nodes of the ins_dag 
    # ins_dag.set_leaf_nodes_value(range(data_config.ins_dag_config.num_leaf_nodes))
    # # propogate the values to the rest of the nodes
    # ins_dag.sync_node_states()
    # abs_dag.sync_node_states()
    # # print the nodes

    # ins_dag.print_nodes()
    # abs_dag.print_nodes()

    # # randomly take a node from ins_dag
    # node = random.choice(list(ins_dag.graph.nodes))
    # math_expr = ins_dag.node_info[node].to_math_expression()
    # print(f"Node: {node}, Math Expression: {math_expr}")

    # # reconstruct the node from the math expression
    # node = DAGWeightedNode.from_math_expression(math_expr, ins_dag.node_info, mod_val=ins_dag.mod_val)
    # node.print_algorithmic_expression()


    # sentence, original_indices = ins_dag.generate_data(shuffle=True)
    # print(f"Sentence: {sentence}")
    # print(f"Original Indices: {original_indices}")

    # # Expand the DAG from the generated sentence
    # # 
    # ins_dag_new = ArithmeticDAG.from_sentence(sentence, original_indices, mod_val=ins_dag.mod_val)
    # ins_dag.print_nodes()