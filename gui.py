import tkinter as tk
from tkinter import filedialog, ttk
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from genalg import Generational_Data, Clique


class CliqueApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clique Problem Visualizer")

        # Create UI components for loading files
        self.load_graph_button = tk.Button(
            root, text="Load Graph File", command=self.load_graph_file
        )
        self.load_graph_button.pack()

        self.load_data_button = tk.Button(
            root, text="Load Data File", command=self.load_data_file
        )
        self.load_data_button.pack()

        self.generation_var = tk.StringVar(root)
        self.generation_dropdown = ttk.Combobox(
            root, textvariable=self.generation_var, state="readonly"
        )
        self.generation_dropdown.bind("<<ComboboxSelected>>", self.on_generation_select)
        self.generation_dropdown.pack()

        # Graph and data placeholders
        self.graph = None
        self.data = None

    def load_graph_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
        if file_path:
            with open(file_path, "rb") as file:
                self.graph = pickle.load(file)
            self.display_graph()

    def load_data_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
        if file_path:
            with open(file_path, "rb") as file:
                self.data = pickle.load(file)
            if "generational" in file_path:
                self.all_cliques = [generation[0] for generational_data in self.data for generation in generational_data.generations]
                        
                self.generation_dropdown["values"] = [
                    f"Generation {i}" for i in range(len(self.all_cliques))
                ]
                self.generation_dropdown.current(0)
                self.on_generation_select(None)
            else:
                self.display_graph(self.data.nodes)

    def on_generation_select(self, event):
        generation_index = self.generation_dropdown.current()
        clique_nodes = self.all_cliques[generation_index].nodes
        self.display_graph(clique_nodes)

    def display_graph(self, clique_nodes=None):
        if self.graph:
            plt.figure(figsize=(8, 6))
            pos = nx.spring_layout(self.graph)
            nx.draw(
                self.graph,
                pos,
                with_labels=True,
                node_color="lightblue",
                edge_color="gray",
            )

            if clique_nodes:
                # Convert set to list for drawing
                clique_nodes_list = list(clique_nodes)
                nx.draw_networkx_nodes(
                    self.graph, pos, nodelist=clique_nodes_list, node_color="orange"
                )

            plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = CliqueApp(root)
    root.mainloop()
