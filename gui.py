import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class CliqueGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Clique Problem Solution Viewer")

        # Open file button
        self.open_button = tk.Button(root, text="Open File", command=self.open_file)
        self.open_button.pack()

        # Dropdown for generation selection
        self.generation_var = tk.StringVar(root)
        self.generation_dropdown = ttk.Combobox(root, textvariable=self.generation_var, state='readonly')
        self.generation_dropdown.bind('<<ComboboxSelected>>', self.on_generation_select)
        self.generation_dropdown.pack()

        # Placeholder for the matplotlib plot
        self.canvas = None

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
        if file_path:
            try:
                with open(file_path, "rb") as file:
                    self.data = pickle.load(file)
                self.populate_generations_dropdown()
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")

    def populate_generations_dropdown(self):
        generations = [f"Generation {i}" for i in range(len(self.data))]
        self.generation_dropdown['values'] = generations
        self.generation_dropdown.current(0)
        self.on_generation_select(None)

    def on_generation_select(self, event):
        generation_index = self.generation_dropdown.current()
        clique, fitness = self.data[generation_index]
        self.display_clique(generation_index, clique, fitness)

    def display_clique(self, generation, clique, fitness):
        fig, ax = plt.subplots()

        # Plotting the number of nodes in the clique (fitness)
        ax.bar(["Clique Size"], [len(clique)])
        ax.set_title(f"Generation {generation}: Clique Size = {len(clique)}")

        # Clear previous canvas
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()

        # Embedding plot in the Tkinter GUI
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

root = tk.Tk()
app = CliqueGUI(root)
root.mainloop()
