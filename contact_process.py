import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import time

class ContactProcess:
    def __init__(self, graph, infection_rate=1.0, recovery_rate=1.0):
        """
        Initialize the Contact Process simulation.
        
        Parameters:
        -----------
        graph : networkx.Graph
            The graph on which the contact process runs
        infection_rate : float
            Rate at which infected nodes infect their neighbors
        recovery_rate : float
            Rate at which infected nodes recover
        """
        self.graph = graph
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.infected = set()  # Set of infected nodes
        self.time = 0.0  # Current time
        self.history = []  # List of (time, num_infected) tuples
        
    def initialize(self, initial_infected=None):
        """
        Initialize the simulation with infected nodes.
        
        Parameters:
        -----------
        initial_infected : list or None
            List of initially infected nodes. If None, infect a random node.
        """
        self.infected.clear()
        self.time = 0.0
        self.history = []
        
        if initial_infected is None:
            # Choose a random node to infect
            initial_infected = [np.random.choice(list(self.graph.nodes()))]
        
        for node in initial_infected:
            self.infected.add(node)
            
        self.history.append((self.time, len(self.infected)))
    
    def step(self):
        """Perform a single step of the contact process."""
        if not self.infected:
            # No infected nodes, process is extinct
            return False
        
        # Calculate total event rate
        total_rate = len(self.infected) * self.recovery_rate  # Recovery events
        for infected_node in self.infected:
            # Count susceptible neighbors for each infected node
            susceptible_neighbors = [n for n in self.graph.neighbors(infected_node) 
                                    if n not in self.infected]
            total_rate += len(susceptible_neighbors) * self.infection_rate
        
        # No possible events
        if total_rate == 0:
            return False
        
        # Time until next event (exponential distribution)
        dt = np.random.exponential(1.0 / total_rate)
        self.time += dt
        
        # Choose an event
        r = np.random.random() * total_rate
        
        # Recovery events
        recovery_threshold = len(self.infected) * self.recovery_rate
        if r < recovery_threshold:
            # Choose a random infected node to recover
            node_to_recover = list(self.infected)[int(r / self.recovery_rate)]
            self.infected.remove(node_to_recover)
        else:
            # Infection events
            r -= recovery_threshold
            
            current_sum = 0
            for infected_node in self.infected:
                susceptible_neighbors = [n for n in self.graph.neighbors(infected_node) 
                                        if n not in self.infected]
                infection_threshold = len(susceptible_neighbors) * self.infection_rate
                if r < current_sum + infection_threshold:
                    # Select the neighbor to infect
                    idx = int((r - current_sum) / self.infection_rate)
                    node_to_infect = susceptible_neighbors[idx]
                    self.infected.add(node_to_infect)
                    break
                current_sum += infection_threshold
        
        # Record the new state
        self.history.append((self.time, len(self.infected)))
        return True
    
    def run(self, max_time=None, max_steps=None):
        """
        Run the simulation until max_time or max_steps is reached.
        
        Parameters:
        -----------
        max_time : float or None
            Maximum simulation time
        max_steps : int or None
            Maximum number of steps
        
        Returns:
        --------
        history : list
            List of (time, num_infected) tuples
        """
        steps = 0
        while True:
            # Check termination conditions
            if max_time is not None and self.time >= max_time:
                break
            if max_steps is not None and steps >= max_steps:
                break
            if not self.infected:  # Process is extinct
                break
                
            # Perform a step
            self.step()
            steps += 1
            
        return self.history

def run_multiple_simulations(graph, num_runs=10, infection_rate=1.0, recovery_rate=1.0, 
                             max_time=10.0, initial_infected=None):
    """
    Run multiple simulations and average the results.
    
    Parameters:
    -----------
    graph : networkx.Graph
        The graph on which the contact process runs
    num_runs : int
        Number of simulations to run
    infection_rate : float
        Rate at which infected nodes infect their neighbors
    recovery_rate : float
        Rate at which infected nodes recover
    max_time : float
        Maximum simulation time
    initial_infected : list or None
        List of initially infected nodes. If None, infect a random node in each run.
    
    Returns:
    --------
    avg_times : numpy.ndarray
        Time points for the averaged curve
    avg_infected : numpy.ndarray
        Average number of infected nodes at each time point
    all_histories : list
        Raw histories from all simulation runs
    """
    cp = ContactProcess(graph, infection_rate, recovery_rate)
    all_histories = []
    
    for i in range(num_runs):
        print(f"Running simulation {i+1}/{num_runs}")
        cp.initialize(initial_infected)
        history = cp.run(max_time=max_time)
        all_histories.append(history)
    
    # Interpolate results to fixed time points for averaging
    time_points = np.linspace(0, max_time, 100)
    interpolated = []
    
    for history in all_histories:
        times, infected = zip(*history)
        times = np.array(times)
        infected = np.array(infected)
        
        # Interpolate this history to the fixed time points
        interp_infected = np.interp(time_points, times, infected)
        interpolated.append(interp_infected)
    
    # Average across all runs
    avg_infected = np.mean(interpolated, axis=0)
    
    return time_points, avg_infected, all_histories

# Example usage
if __name__ == "__main__":
    # Create a graph (change this to your desired graph)
    # Example: 2D lattice
    N = 5  # 20x20 grid
    G = nx.grid_2d_graph(N, N)
    
    # Relabel nodes to integers for simplicity
    G = nx.convert_node_labels_to_integers(G)
    
    # Parameters
    infection_rate = 1
    recovery_rate = 1.0
    max_time = 20000.0
    num_runs = 10
    
    # Run the simulations
    start_time = time.time()
    times, avg_infected, all_histories = run_multiple_simulations(
        G, num_runs=num_runs, infection_rate=infection_rate, 
        recovery_rate=recovery_rate, max_time=max_time
    )
    end_time = time.time()
    
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Plot the average infection curve
    plt.figure(figsize=(10, 6))
    
    # Plot individual runs (lightly)
    for i, history in enumerate(all_histories):
        t, inf = zip(*history)
        plt.plot(t, inf, alpha=0.2, color='gray', linewidth=1)
    
    # Plot the average
    plt.plot(times, avg_infected, 'r-', linewidth=2, label='Average')
    
    plt.title(f'Contact Process on {N}x{N} Grid (λ={infection_rate}, δ={recovery_rate})')
    plt.xlabel('Time')
    plt.ylabel('Number of Infected Nodes')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Also plot the fraction of infected nodes
    plt.figure(figsize=(10, 6))
    plt.plot(times, avg_infected / G.number_of_nodes(), 'g-', linewidth=2)
    plt.title(f'Contact Process - Fraction of Infected Nodes (λ={infection_rate}, δ={recovery_rate})')
    plt.xlabel('Time')
    plt.ylabel('Fraction of Infected Nodes')
    plt.grid(True, alpha=0.3)
    
    plt.show() 