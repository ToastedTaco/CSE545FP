import pickle
from genalg import Generational_Data, Clique
import os

if __name__ == "__main__":
    filepath = f"data_files{os.path.sep}random_graph_50_nodes_wisdom_of_crowds.pkl"
    
    with open(filepath, "rb") as file:
        data = pickle.load(file)
        
    print(data.nodes)