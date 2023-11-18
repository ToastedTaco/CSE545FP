import tkinter as tk
from tkinter import ttk, filedialog
import pickle
import matplotlib.pyplot as plt
import os
from genalg import Generational_Data, Clique
import networkx as nx

# Define global variables
base_graph = None
generations = None
all_cliques = None
all_cliques_fitness = None
woc_clique = None
plot_image = None
num_highlighted_nodes = None
selected_generation = None


# Function to select a data file using a file dialog
def select_data_file():
    global data_filename
    new_data_filename = filedialog.askopenfilename(
        filetypes=[("Pickle Files", "*.pkl")]
    )
    if new_data_filename:
        print(f"Selected data file: {new_data_filename}")
        data_filename = new_data_filename  # Update the data_filename


# Function to create and display the GUI
def display_gui():
    global base_graph, generations, all_cliques, all_cliques_fitness, woc_clique, plot_image, num_highlighted_nodes, selected_generation

    # Load data from a data file
    try:
        with open(data_filename, "rb") as import_file:
            imported_data = pickle.load(import_file)

        base_graph = imported_data["base_graph"]
        all_cliques = imported_data["all_cliques"]
        all_cliques_fitness = [None] * len(all_cliques)
        generations = imported_data["generations"]
        woc_clique = imported_data["woc_clique"]
        
        for i in range(len(all_cliques)):
            all_cliques_fitness[i] = len(all_cliques[i].nodes)

        # Create the main Tkinter window
        root = tk.Tk()
        root.title("Genetic Algorithm TSP")
        root.geometry("1200x800")  # Set the window size

        # Create a menu bar
        menubar = tk.Menu(root)
        root.config(menu=menubar)

        # Create a "File" menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)

        # Add an "Open Data File" option to the "File" menu
        file_menu.add_command(label="Open Data File", command=select_data_file)

        # Add a separator
        file_menu.add_separator()

        # Add an "Exit" option to the "File" menu
        file_menu.add_command(label="Exit", command=root.quit)

        # Create a canvas for displaying graphs
        canvas = tk.Canvas(root)
        canvas.pack(fill=tk.BOTH, expand=True)  # Dynamically scale the canvas

        num_highlighted_nodes = 0
        default_gen = 0

        # Create a generation selection dropdown
        generation_var = tk.IntVar()
        generation_label = ttk.Label(root, text="Select Generation:")
        generation_label.pack()
        generation_dropdown = ttk.Combobox(
            root, textvariable=generation_var, state="readonly"
        )
        generation_dropdown.pack()
        generation_dropdown.set(default_gen)  # Default to the best generation

        # Function to update the generation dropdown based on the selected thread
        def update_generation_dropdown(*args):
            generation_dropdown["values"] = generations

        # Function to update the data displayed in the GUI
        def update_data():
            global base_graph, generations, all_cliques, all_cliques_fitness, woc_clique, plot_image, num_highlighted_nodes, selected_generation
            try:
                with open(data_filename, "rb") as import_file:
                    imported_data = pickle.load(import_file)

                base_graph = imported_data["base_graph"]
                all_cliques = imported_data["all_cliques"]
                all_cliques_fitness = [None]*len(all_cliques)
                generations = imported_data["generations"]
                woc_clique = imported_data["woc_clique"]
                
                for i in range(len(all_cliques)):
                    all_cliques_fitness[i] = len(all_cliques[i].nodes)

                default_gen = 0
                default_cost = 0

                generation_var.set(
                    default_gen
                )  # Update the existing generation dropdown
                update_cost_label(default_cost)
                update_graph()

            except FileNotFoundError:
                print(
                    "Data file not found. Please run the genetic algorithm script to generate data."
                )

        # Function to update the displayed graph based on user selections
        def update_graph():
            selected_generation = int(generation_var.get())
            if base_graph:
                if all_cliques is not None and all_cliques_fitness is not None:
                    # Update the graph based on user selections
                    if plot_var.get() == "Path":
                        # Plot the path graph
                        canvas.delete("all")
                        plt.figure(figsize=(8, 6))
                        pos = nx.spring_layout(base_graph)
                        nx.draw(
                            base_graph,
                            pos,
                            with_labels=True,
                            node_color="lightblue",
                            edge_color="gray",
                        )

                        clique_nodes = all_cliques[selected_generation].nodes
                        if clique_nodes:
                            # Convert set to list for drawing
                            clique_nodes_list = list(clique_nodes)
                            nx.draw_networkx_nodes(
                                base_graph,
                                pos,
                                nodelist=clique_nodes_list,
                                node_color="orange",
                            )

                        plt.title(
                            f"{os.path.basename(data_filename)}, Gen: {selected_generation} Nodes"
                        )
                        cost = all_cliques_fitness[selected_generation]
                        plt.suptitle(f"Performance: {cost}")
                        plt.xlabel("X")
                        plt.ylabel("Y")
                        plt.grid(True)
                        plt.savefig(
                            "temp_plot.png"
                        )  # Save the plot to a temporary file
                        plot_image = tk.PhotoImage(file="temp_plot.png")
                        canvas.create_image(0, 0, anchor=tk.NW, image=plot_image)
                        canvas.image = plot_image
                        update_cost_label(cost)

                    elif plot_var.get() == "Performance":
                        # Plot the cost vs. generation graph
                        all_cliques_fitness_sorted = all_cliques_fitness
                        all_cliques_fitness_sorted.sort()
                        
                        canvas.delete("all")
                        plt.figure(figsize=(8, 6))
                        
                        plt.plot(generations, all_cliques_fitness_sorted)
                        plt.title(
                            f"{os.path.basename(data_filename)}, Performance vs. Generation"
                        )
                        plt.xlabel("Generation")
                        plt.ylabel("Performance")
                        plt.grid(True)
                        plt.savefig(
                            "temp_plot.png"
                        )  # Save the plot to a temporary file
                        plot_image = tk.PhotoImage(file="temp_plot.png")
                        canvas.create_image(0, 0, anchor=tk.NW, image=plot_image)
                        canvas.image = plot_image
                        generation_dropdown.set(0)
                        update_cost_label(max(all_cliques_fitness))

                    elif plot_var.get() == "WoC":
                        # Plot the path graph
                        canvas.delete("all")
                        plt.figure(figsize=(8, 6))
                        pos = nx.spring_layout(base_graph)
                        nx.draw(
                            base_graph,
                            pos,
                            with_labels=True,
                            node_color="lightblue",
                            edge_color="gray",
                        )

                        clique_nodes = woc_clique.nodes
                        if clique_nodes:
                            # Convert set to list for drawing
                            clique_nodes_list = list(clique_nodes)
                            nx.draw_networkx_nodes(
                                base_graph,
                                pos,
                                nodelist=clique_nodes_list,
                                node_color="orange",
                            )

                        plt.title(
                            f"{os.path.basename(data_filename)}, WoC Nodes"
                        )
                        cost = len(woc_clique.nodes)
                        plt.suptitle(f"Performance: {cost}")
                        plt.xlabel("X")
                        plt.ylabel("Y")
                        plt.grid(True)
                        plt.savefig(
                            "temp_plot.png"
                        )  # Save the plot to a temporary file
                        plot_image = tk.PhotoImage(file="temp_plot.png")
                        canvas.create_image(0, 0, anchor=tk.NW, image=plot_image)
                        canvas.image = plot_image
                        update_cost_label(cost)
                        
                    elif plot_var.get() == "Base_Graph":
                        # Plot the path graph
                        canvas.delete("all")
                        plt.figure(figsize=(8, 6))
                        pos = nx.spring_layout(base_graph)
                        nx.draw(
                            base_graph,
                            pos,
                            with_labels=True,
                            node_color="lightblue",
                            edge_color="gray",
                        )

                        plt.title(
                            f"{os.path.basename(data_filename)}, Base Graph"
                        )
                        cost = 0
                        plt.xlabel("X")
                        plt.ylabel("Y")
                        plt.grid(True)
                        plt.savefig(
                            "temp_plot.png"
                        )  # Save the plot to a temporary file
                        plot_image = tk.PhotoImage(file="temp_plot.png")
                        canvas.create_image(0, 0, anchor=tk.NW, image=plot_image)
                        canvas.image = plot_image
                        update_cost_label(cost)

        # Function to update the cost label based on user selections
        def update_cost_label(cost):
            cost_label.config(text=f"Performance: {cost}")

        # Create a label for displaying the cost
        cost_label = ttk.Label(root, text="")
        cost_label.pack()

        # Create radio buttons for selecting the plot type
        plot_var = tk.StringVar()
        plot_var.set("Base_Graph")  # Default to Path plot
        base_button = ttk.Radiobutton(
            root, text="Base_Graph", variable=plot_var, value="Base_Graph"
        )
        base_button.pack()  
        
        path_button = ttk.Radiobutton(
            root, text="Path", variable=plot_var, value="Path"
        )
        path_button.pack()

        cost_button = ttk.Radiobutton(
            root, text="Performance vs. Generation", variable=plot_var, value="Performance"
        )
        cost_button.pack()

        woc_button = ttk.Radiobutton(
            root, text="Wisdom of the Crowds", variable=plot_var, value="WoC"
        )
        woc_button.pack()

        # Create a button to update the graph
        update_button = ttk.Button(root, text="Update Graph", command=update_graph)
        update_button.pack()

        # Create a button to update the data
        update_button = ttk.Button(root, text="Update Data", command=update_data)
        update_button.pack()

        # Initial update of the graph
        update_graph()
        update_generation_dropdown()

        # Start the Tkinter main loop
        root.mainloop()

    except FileNotFoundError:
        print(
            "Data file not found. Please run the genetic algorithm script to generate data."
        )


if __name__ == "__main__":
    data_filename = f"data_files{os.path.sep}random_graph_20_datapack.pkl"  # Set the initial data file here
    display_gui()
