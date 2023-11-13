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
            # Check if each neighbor of the node is also in the clique
            if not all(neighbor in self.nodes for neighbor in graph.adjacency_list[node]):
                return 0  # If any neighbor is not in the clique, it's not a valid clique

        return len(self.nodes)  # Valid clique: return its size

class Population:
    def __init__(self, population_size, graph: Graph, crossover_type="intersection"):
        self.population_size = population_size
        self.graph = graph
        self.crossover_type = crossover_type
        self.population = [self.create_random_clique() for _ in range(population_size)]

    def create_random_clique(self):
        nodes = set(
            random.sample(
                list(self.graph.adjacency_list.keys()),
                k=random.randint(2, len(self.graph.adjacency_list)),
            )
        )
        return Clique(nodes)

    def evolve(self, mutation_rate: float, generation_data):
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(self.population, 2)
            child = self.crossover(parent1, parent2)
            self.mutate(child, mutation_rate)
            new_population.append(child)
        self.population = new_population

        # Record fitness for each clique in the population
        generation_fitness = [clique.calculate_fitness(self.graph) for clique in self.population]
        generation_data.append(generation_fitness)

        # Print the status of the current generation
        print(f"Generation {len(generation_data)} - Best Fitness: {max(generation_fitness)}")

    def crossover(self, parent1, parent2):
        if self.crossover_type == 'intersection':
            child_nodes = parent1.nodes.intersection(parent2.nodes)
        elif self.crossover_type == 'union':
            child_nodes = parent1.nodes.union(parent2.nodes)
        return Clique(child_nodes)

    def mutate(self, clique, mutation_rate):
        if random.random() < mutation_rate:
            if clique.nodes and random.random() < 0.5:
                clique.nodes.remove(random.choice(list(clique.nodes)))
            else:
                potential_nodes = set(self.graph.adjacency_list.keys()) - clique.nodes
                if potential_nodes:
                    clique.nodes.add(random.choice(list(potential_nodes)))

    def get_best_clique(self):
        return max(self.population, key=lambda clique: clique.calculate_fitness(self.graph))

def aggregate_runs_data(runs_data):
    cliques = [tuple(sorted(data[0])) for data in runs_data]
    most_common_clique = max(set(cliques), key=cliques.count)
    return set(most_common_clique)

def run_genetic_algorithm(graph, population_size=100, mutation_rate=0.01, generations=500, num_runs=10):
    all_runs_data = []
    for _ in range(num_runs):
        population = Population(population_size, graph)
        generational_data = []
        for _ in range(generations):
            population.evolve(mutation_rate, generational_data)
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
            runs_data = run_genetic_algorithm(graph)
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
