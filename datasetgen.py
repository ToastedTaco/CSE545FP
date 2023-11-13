import networkx as nx
import pickle
import random
import os

# Function to generate a random graph
def generate_random_graph(n, p):
    # Adjusting probability based on the number of nodes
    adjusted_p = p / (n ** 0.5)
    return nx.erdos_renyi_graph(n, adjusted_p)

# Function to save a graph to a pickle file
def save_graph_to_pickle(graph, filename):
    """
    Saves a graph object to a pickle file.
    
    :param graph: A NetworkX graph.
    :param filename: Filename for the pickle file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(graph, f)

# Generate and save graphs of different sizes
node_counts = [20, 50, 100, 500]
dump_folder = "uploads"
for node_count in node_counts:
    graph = generate_random_graph(node_count, 0.5)  # Adjust probability as needed
    filename = f'{dump_folder}{os.path.sep}random_graph_{node_count}_nodes.pkl'
    save_graph_to_pickle(graph, filename)
    print(f"Graph with {node_count} nodes saved as {filename}")
