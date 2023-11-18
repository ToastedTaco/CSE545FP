import pickle
import os
from genalg import Generational_Data, Clique

graph_file = r"C:\Users\lambi\OneDrive\Documents\GitHub\CSE545FP\uploads\random_graph_20_nodes.pkl"
generational_file = r"C:\Users\lambi\OneDrive\Documents\GitHub\CSE545FP\data_files\random_graph_20_nodes_generational_data.pkl"
woc_file = r"C:\Users\lambi\OneDrive\Documents\GitHub\CSE545FP\data_files\random_graph_20_nodes_wisdom_of_crowds.pkl"

dump_folder = "data_files"


with open(graph_file, "rb") as file:
    graph_data = pickle.load(file)

with open(generational_file, "rb") as file:
    generational_data = pickle.load(file)

with open(woc_file, "rb") as file:
    woc_data = pickle.load(file)


all_cliques = [
    generation[0]
    for generational_data in generational_data
    for generation in generational_data.generations
]

generations = [i for i in range(len(all_cliques))]

all_cliques_fitness = [
    generation[1]
    for generational_data in generational_data
    for generation in generational_data.generations
]

all_cliques_fitness.sort()

data_pack = {
    "base_graph": graph_data,
    "all_cliques": all_cliques,
    "all_cliques_fitness": all_cliques_fitness,
    "generations": generations,
    "woc_clique": woc_data,
}

data_pack_filename = f"{dump_folder}{os.path.sep}random_graph_20_datapack.pkl"

with open(data_pack_filename, "wb") as file:
    pickle.dump(data_pack, file)
