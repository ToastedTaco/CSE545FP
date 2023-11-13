import os
import random
import pickle
import networkx as nx

class Graph:
    def __init__(self, edge_list):
        self.edge_list = edge_list
        self.adjacency_list = self.create_adjacency_list()

    def create_adjacency_list(self):
        adjacency_list = {}
        for edge in self.edge_list:
            adjacency_list.setdefault(edge[0], []).append(edge[1])
            adjacency_list.setdefault(edge[1], []).append(edge[0])
        return adjacency_list

def load_graph_from_pickle(filename):
    with open(filename, "rb") as f:
        graph = pickle.load(f)
    return Graph([(u, v) for u, v in graph.edges()])

class Clique:
    def __init__(self, nodes: set):
        self.nodes = nodes

    def calculate_fitness(self, graph: Graph) -> float:
        if not self.nodes:
            return 0  # An empty set of nodes is not a valid clique

        for node in self.nodes:
            if len(self.nodes) == 1:
                break
            # Check if each neighbor of the node is also in the clique
            adj = graph.adjacency_list[node]
            nodes = self.nodes
            if not all(neighbor in self.nodes for neighbor in graph.adjacency_list[node]):
                return 0  # If any neighbor is not in the clique, it's not a valid clique

        return len(self.nodes)  # Valid clique: return its size

class Population:
    def __init__(self, population_size, graph: Graph, elitism_rate=0.1):
        self.population_size = population_size
        self.graph = graph
        self.elitism_rate = elitism_rate
        self.population = [self.create_random_clique() for _ in range(population_size)]
        self.best_fitness = 0
        self.best_clique = None

    def create_random_clique(self):
        node = random.choice(list(self.graph.adjacency_list.keys()))
        return Clique({node})

    def evolve(self, file, run, generation, mutation_rate: float):
        # Perform elitism to carry over a percentage of the best cliques to the new population
        new_population = self.elitism()

        # Fill the rest of the new population by selection, crossover, and mutation
        while len(new_population) < self.population_size:
            # Selection through a tournament approach
            parent1, parent2 = self.tournament_selection(), self.tournament_selection()
            
            # Crossover to produce a child clique from two parents
            child_clique = self.crossover(parent1, parent2)
            
            # Mutation with a certain probability to introduce variations
            if random.random() < mutation_rate:
                child_clique = self.mutate(child_clique)
            
            # Add the new child clique to the new population
            new_population.append(child_clique)

        # Replace the old population with the new population
        self.population = new_population
        
        # Update the best clique and its fitness if the current generation provides an improvement
        current_best_clique = self.get_best_clique()
        current_best_fitness = current_best_clique.calculate_fitness(self.graph)
        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_clique = current_best_clique
        
        # Print details about the current generation's progress
        print(f"File: {file}, Run: {run}, Generation: {generation}, Best Fitness: {self.best_fitness}")

    def elitism(self):
        sorted_population = sorted(self.population, key=lambda clique: clique.calculate_fitness(self.graph), reverse=True)
        retain_length = int(len(sorted_population) * self.elitism_rate)
        return sorted_population[:retain_length]

    def tournament_selection(self):
        tournament = random.sample(self.population, k=5)
        fittest = max(tournament, key=lambda clique: clique.calculate_fitness(self.graph))
        return fittest

    def crossover(self, parent1, parent2):
        # Implement a more sophisticated crossover that maintains graph connectivity
        common_neighbors = set(parent1.nodes).intersection(parent2.nodes)
        if not common_neighbors:
            common_neighbors = {random.choice(list(self.graph.adjacency_list.keys()))}
        # Attempt to add more nodes that are neighbors to this set
        potential_nodes = set().union(*(self.graph.adjacency_list[node] for node in common_neighbors))
        # Ensure potential nodes are connected to all in the clique
        new_nodes = {node for node in potential_nodes if all(neighbor in potential_nodes or neighbor in common_neighbors for neighbor in self.graph.adjacency_list[node])}
        return Clique(common_neighbors.union(new_nodes))

    def mutate(self, clique):
        if clique.nodes and random.random() < 0.5:
            clique.nodes.pop()
        else:
            potential_nodes = set(self.graph.adjacency_list.keys()) - clique.nodes
            clique.nodes.add(random.choice(list(potential_nodes)))
        return clique

    def update_best(self):
        current_best = self.get_best_clique()
        current_fitness = current_best.calculate_fitness(self.graph)
        if current_fitness > self.best_fitness:
            self.best_fitness = current_fitness
            self.best_clique = current_best

    def get_best_clique(self):
        return max(self.population, key=lambda clique: clique.calculate_fitness(self.graph))

def aggregate_runs_data(runs_data):
    cliques = [tuple(sorted(data[0])) for data in runs_data]
    most_common_clique = max(set(cliques), key=cliques.count)
    return set(most_common_clique)

def run_genetic_algorithm(file, graph, population_size=100, mutation_rate=0.5, generations=500, num_runs=10):
    all_runs_data = []
    for run in range(num_runs):
        population = Population(population_size, graph)
        for generation in range(generations):
            population.evolve(file, run, generation, mutation_rate)
        best_clique = population.get_best_clique()
        run_result = (best_clique.nodes, best_clique.calculate_fitness(graph))
        all_runs_data.append(run_result)
    return all_runs_data

def save_to_pickle(data, file_name, save_folder):
    save_path = os.path.join(save_folder, file_name)
    with open(save_path, "wb") as file:
        pickle.dump(data, file)

def process_folder(input_folder, output_folder):
    for file in os.listdir(input_folder):
        if file.endswith(".pkl"):
            full_path = os.path.join(input_folder, file)
            graph = load_graph_from_pickle(full_path)
            runs_data = run_genetic_algorithm(file, graph)
            generational_file = file.replace(".pkl", "_generational_data.pkl")
            wisdom_file = file.replace(".pkl", "_wisdom_of_crowds.pkl")
            save_to_pickle(runs_data, generational_file, output_folder)
            wisdom_of_crowds_clique = aggregate_runs_data(runs_data)
            save_to_pickle(wisdom_of_crowds_clique, wisdom_file, output_folder)

if __name__ == "__main__":
    uploads_folder = "uploads"  # Folder with input datasets
    results_folder = "data_files"  # Folder to save results
    os.makedirs(results_folder, exist_ok=True)  # Create the results folder if it doesn't exist
    process_folder(uploads_folder, results_folder)
