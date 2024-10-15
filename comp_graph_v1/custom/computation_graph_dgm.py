import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx

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

class ComputationGraphDGM():
    def __init__(self, var_vocab, mod_val, function_map):
        """A class for generating examples of computation graphs and corresponding "language modeling" prompts.


        Parameters
        ----------
        var_vocab : list[str]
            list of variable names
        mod_val : int
            base for modular arithmetic in computation graph
        function_map : dict[str, function]
            dictionary mapping function names to functions that take a list of integers and return an integer
        """

        # NOTE: for now, functions are assumed to accept any number of arguments (except for leaf assignmnet)
        # TODO: generalize to allow each type of function to accept a specific number of arguments

        self.var_vocab = var_vocab

        self.mod_val = mod_val
        self.numeric_vocab = [str(v) for v in range(mod_val)]

        self.function_map = function_map
        self.func_vocab = list(function_map.keys())

        self.operand_vocab = [',', '(', ')']

        self.equal_token = '->'
        self.query_token = '<query>'
        self.answer_token = '<answer>'
        self.eq_sep_token = '<eq_sep>'
        self.sep_token = '<sep>'


        self.vocab = self.var_vocab + self.numeric_vocab + self.func_vocab + self.operand_vocab
        self.vocab += [self.equal_token, self.query_token, self.answer_token, self.eq_sep_token, self.sep_token]
        self.tokenizer = Tokenizer(self.vocab)

    def sample_example(self, n_vars, func_degree, query_var='random', verbose=False):
        # randomly sample a computation graph example; return a dictionary with keys: prompt, edges, func_annotations, var_top_order, node_vals, query_prompt
        sample = self._sample_computation_graph_example(n_vars, func_degree, verbose=verbose)

        # solve computation and add `node_vals` key to `sample` dict`. `node_vals` is a dictionary of solved node values (i.e., map each var to its value)
        self._solve_sample(sample, verbose=verbose)

        # create query prompt and add `query_prompt` key to `sample` dict. `query_prompt` is a list of tokens
        # query prompt is a copy of the prompt with a randomly sampled query about the graph and answer appended
        self._create_query_prompt(sample, query_var=query_var, verbose=verbose)

        return sample

    def _sample_computation_graph_example(self, n_vars, func_degree, verbose=False):

        # first, sample a topological order of variables for computation DAG
        var_top_order = list(np.random.choice(self.var_vocab, size=n_vars, replace=False))

        edges = [] # for storing edges in DAG
        func_annotations = dict() # for storing function annotations for each variable
        prompt = [] # for storing prompt

        if verbose:
            print(f'topological order of variables in DAG: {var_top_order}')

        for idx, var in enumerate(var_top_order):
            # check if leaf node
            if idx < func_degree: # for now, first func_degree variables are leaf nodes (because too few possible children)
                var_val = random.choice(self.numeric_vocab)
                edges.append((var_val, var))
                func_annotations[var] = "leafValueAssignment"
                prompt += [var_val, self.equal_token, var, self.eq_sep_token]
                if verbose:
                    print(f'{var} <- {var_val}')
            # if not leaf node, children are randomly chosen from preceeding variables (wrt topological order)
            # node is a randomly-selected function of its children
            else:
                # randomly sample children from preceeding variables
                children = np.random.choice(var_top_order[:idx], size=func_degree, replace=False).tolist()

                # randomly sample function
                func = np.random.choice(self.func_vocab)
                func_annotations[var] = func

                for child in children:
                    edges.append((child, var))

                prompt += [func, '('] + list(','.join(children)) + [')'] + [self.equal_token, var, self.eq_sep_token]

                if verbose:
                    print(f'{var} <- {func}({", ".join(children)})')

        if verbose:
            print()
            print(f"prompt: {' '.join(prompt)}")

        return dict(prompt=prompt, edges=edges, func_annotations=func_annotations, var_top_order=var_top_order)

    def _solve_sample(self, sample, verbose=False):
        """add `node_vals` key to `sample` dict`. `node_vals` is a dictionary of solved node values (i.e., map each var to its value)"""
        edges = sample['edges']
        func_annotations = sample['func_annotations']
        var_top_order = sample['var_top_order']

        func_map = self.function_map.copy()
        func_map['leafValueAssignment'] = leafValueAssignment # add leafValueAssignment function to function_map


        G = nx.from_edgelist(edges, create_using=nx.DiGraph)
        nx.set_node_attributes(G, func_annotations, 'func')
        node_vals = {node: int(node) if node in self.numeric_vocab else None for node in G.nodes()}
        nx.set_node_attributes(G, node_vals, 'val')

        if verbose:
            print()
            print('Solving computation graph...')
            print('initializing node values:')
            print(node_vals)
            print()
            print('now iterating in topological order and solving for the value of each node...')
            print()

        for var in var_top_order:
            children = list(G.predecessors(var)) # NOTE: unintuitively (for me), predecessors are called children in computation graphs
            par_vals = [node_vals[child] for child in children]
            if verbose:
                print(f'Children({var}) = {[f"{p}={pv}" for p, pv in zip(children, par_vals)]}')
            func_name = func_annotations[var]
            func = func_map[func_name]
            node_vals[var] = func(par_vals)
            if verbose:
                print(f'{var} <- {func_name}({[node_vals[child] for child in children]}) = {node_vals[var]}')
                print()

        sample['node_vals'] = node_vals

    def _create_query_prompt(self, sample, query_var='random', verbose=False):
        query_prompt = sample['prompt'].copy()
        node_vals = sample['node_vals']
        var_top_order = sample['var_top_order']

        if query_var == 'last':
            query_var = var_top_order[-1] # either last var in computation graph or randomly sample from vars
        elif query_var == 'random':
            query_var = np.random.choice(var_top_order)
        else:
            raise ValueError("query_var must be 'last' or 'random'")

        query_prompt += [self.query_token, query_var, self.answer_token, str(node_vals[query_var])]

        if verbose:
            print(' '.join(query_prompt))

        sample['query_var'] = query_var
        sample['query_prompt'] = query_prompt

        sample['tokenized_query_prompt'] = self.tokenizer.encode_tokens(query_prompt)

def leafValueAssignment(children):
    if len(children) == 1:
        return children[0]
    else:
        raise ValueError("leafValueAssignment function should have exactly one child")

def draw_topological_order(sample, numeric_vocab, query_node=None, rad=-0.8, figsize=(8,4), orientation='horizontal'):
    """draw DAG according to topological order"""

    G = nx.from_edgelist(sample['edges'], create_using=nx.DiGraph)

    node_options = dict(edgecolors='tab:gray', node_size=800, alpha=0.8)
    edge_options = dict(edge_color='tab:gray', alpha=0.9, width=1.5,)

    # red if ancestor, blue if query node, black otherwise
    if query_node is not None:
        # ancestors of query node
        ancestors = nx.ancestors(G, query_node)
        node_colors = ['tab:red' if node in ancestors else 'tab:green' if node == query_node else 'tab:blue' for node in G.nodes()]
    else:
        node_colors = 'tab:blue'

    top_order = sample['var_top_order']
    numeric_nodes = [node for node in G.nodes() if node in numeric_vocab]
    top_order = numeric_nodes + top_order

    if orientation == 'horizontal':
        pos = {node:(top_order.index(node),0) for node in G.nodes()}
    else:
        pos = {node:(0,top_order.index(node)) for node in G.nodes()}

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(G, node_color=node_colors, pos=pos, **node_options)

    # list doesn't work for some reason...
    # connection_styles = [f'arc3,rad={rad * (-1)**top_order.index(source)}' for source, target in G.edges()]

    for edge in G.edges():
        source, target = edge
        nx.draw_networkx_edges(G, pos=pos, edgelist=[edge], connectionstyle=f'arc3,rad={rad*(-1)**top_order.index(source)}',
            **edge_options, node_size=node_options['node_size'])

    nx.draw_networkx_labels(G, pos=pos, font_color='whitesmoke')
    plt.box(False)
    return fig, ax


class ADD:
    def __init__(self, mod_val) -> None:
        self.mod_val = mod_val
    def __call__(self, x, y):
        return (x + y) % self.mod_val
    @property
    def __name__(self):
        return "ADD"

class MUL:
    def __init__(self, mod_val) -> None:
        self.mod_val = mod_val
    def __call__(self, x, y):
        return (x * y) % self.mod_val
    @property
    def __name__(self):
        return "MUL"
    
class EQUL:
    def __init__(self, mod_val) -> None:
        self.mod_val = mod_val
    def __call__(self, x, y):
        return y % self.mod_val
    @property
    def __name__(self):
        return "EQUL"

class DAGWeightedNode:
    def __init__(self, name, weight=None):
        """
        Initialize a DAG node with a name and an optional weight.
        
        Parameters:
        - name: the name of the node
        - weight: the weight of the node (default is None)
        """
        self.name = name
        self.weight = weight if weight is not None else random.uniform(1.0, 10.0)
        self.fan_in = []  # Stores the fan-in nodes and the applied functions

    def add_fan_in(self, parent_node, func):
        """
        Add a fan-in node with the function used to combine the values.
        
        Parameters:
        - parent_node: the DAGWeightedNode instance of the parent node
        - func: the function used to combine the parent node's value with the current value
        """
        self.fan_in.append((parent_node, func))

    def compute_value(self):
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
                
    def print_algorithmic_expression(self):
        """
        Print an algorithmic expression that represents how the fan-in is calculated.
        """
        if not self.fan_in:
            print(f"{self.name} = {self.weight:.2f} / No fan-in")
        else:
            value = 0
            expression = ""
            for parent_node, func in self.fan_in:
                operation = func.__name__ if hasattr(func, '__name__') else 'func'
                expression += f" {operation} ({parent_node.name})"
                value = func.__call__(value, parent_node.weight)
            expression = f"{self.name} = {self.weight:.2f} / {value} <-" + expression
            print(expression)

class AlgorithmicDAG:
    def __init__(self, vocab, min_fan_in_deg=1, max_fan_in_deg=3, func_vocab=None, mod_val=19):
        """
        Initialize a Directed Acyclic Graph (DAG) in dictionary order, node by node.
        
        Parameters:
        - vocab: list of nodes (e.g., ['a', 'b', 'c', 'd'])
        - min_fan_in_deg: minimum number of incoming edges per node (must be at least 0)
        - max_fan_in_deg: maximum number of incoming edges per node (must be at least 0)
        - func_vocab: list of functions to be used for combining fan-in nodes
        """
        self.mod_val = mod_val
        self.vocab = vocab
        self.min_fan_in_deg = min_fan_in_deg
        self.max_fan_in_deg = max_fan_in_deg
        self.func_vocab = func_vocab if func_vocab else [ADD, MUL]  # Default to addition and multiplication
        self.graph = self._generate_random_dag()
        self.node_info = {node: DAGWeightedNode(node) for node in self.vocab}  # Create DAGWeightedNode instances for each node
        self._assign_node_weights()
        self._init_fan_in_method()
        

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
            if idx > 0:
                # Previous nodes available for connecting
                possible_parents = self.vocab[:idx]
                
                # Determine the number of incoming edges (fan-in degree)
                min_deg = min(self.min_fan_in_deg, idx)  # Ensure it does not exceed the number of available nodes
                max_deg = min(self.max_fan_in_deg, idx)  # Ensure it does not exceed the number of available nodes
                
                # Randomly select the number of parents within bounds
                num_parents = random.randint(min_deg, max_deg)
                
                # Randomly select parent nodes and add edges
                parents = random.sample(possible_parents, num_parents)
                for parent in parents:
                    G.add_edge(parent, node)

        return G

    def _assign_node_weights(self):
        """
        Assign a random weight to each node in the graph.
        """
        for node in self.graph.nodes:
            self.node_info[node].weight = random.uniform(1.0, 10.0)

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

    def sync_node_values(self):
        """
        Iteratively apply each node's compute_value() method to sync all the values in the graph.
        """
        for node in nx.topological_sort(self.graph):
            self.node_info[node].compute_value()
            
    def draw(self):
        """
        Draw the DAG using matplotlib, with node weights displayed.
        """
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.graph)
        node_labels = {node: f"{node}\n(weight={self.node_info[node].weight:.2f})" for node in self.graph.nodes}
        nx.draw(self.graph, pos, with_labels=True, labels=node_labels, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
        plt.title("Directed Acyclic Graph (DAG) with Node Weights")
        plt.show()

# Example usage:
abstract_vocab = ['a', 'b', 'c', 'd', 'e', 'f']
min_fan_in_deg = 1
max_fan_in_deg = 3
func_vocab = [ADD, MUL]  # Example functions

# Create a AlgorithmicDAG instance
dag_instance = AlgorithmicDAG(abstract_vocab, min_fan_in_deg, max_fan_in_deg, func_vocab)

# Sync the values in the graph
dag_instance.sync_node_values()

# Draw the generated DAG
dag_instance.draw()

# Print node information
for node in dag_instance.node_info.values():
    print(f"Node: {node.name}, Weight: {node.weight}, Fan-in: {[f'({parent.name}, {func.__name__})' for parent, func in node.fan_in]}")
    node.print_algorithmic_expression()