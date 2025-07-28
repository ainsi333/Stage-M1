import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import networkx as nx

class DynamicContactProcess:
    def __init__(self, n, p, lambda_val, nu, max_time, initial_infected=None):
        """
        Initialize dynamic contact process simulation.
        
        Args:
            n (int): Number of nodes in the graph
            p (float): Probability parameter for dynamic percolation (in [0,1])
            lambda_val (float): Infection rate for the contact process
            nu (float): Rate parameter for edge dynamics
            max_time (float): Maximum simulation time
            initial_infected (list, optional): List of initially infected nodes
        """
        self.n = n
        self.p = p
        self.lambda_val = lambda_val
        self.nu = nu
        self.max_time = max_time
        
        # Initialize state variables
        self.nodes = list(range(n))
        self.edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # Initialize node states (0 = healthy, 1 = infected)
        self.node_states = np.zeros(n, dtype=int)
        if initial_infected is not None:
            for node in initial_infected:
                self.node_states[node] = 1
        elif n > 0:
            # If no initial infected specified, infect node 0
            self.node_states[0] = 1
            
        # Initialize edge states (0 = closed, 1 = open)
        self.edge_states = np.zeros(len(self.edges), dtype=int)
        # Randomly open edges according to initial probability p
        for i in range(len(self.edges)):
            if np.random.random() < p:
                self.edge_states[i] = 1
        
        # History of events for visualization
        self.time_events = []
        self.node_states_history = []
        self.edge_states_history = []
        self.edge_indices = {edge: i for i, edge in enumerate(self.edges)}
        
        # Record initial state
        self.time_events.append(0)
        self.node_states_history.append(self.node_states.copy())
        self.edge_states_history.append(self.edge_states.copy())
    
    def get_neighbors(self, node):
        """Get all open neighbors of a node."""
        neighbors = []
        for i, edge in enumerate(self.edges):
            if self.edge_states[i] == 1:  # Only consider open edges
                if edge[0] == node:
                    neighbors.append(edge[1])
                elif edge[1] == node:
                    neighbors.append(edge[0])
        return neighbors
    
    def simulate(self):
        """Run the simulation until max_time."""
        current_time = 0
        
        while current_time < self.max_time:
            # Compute rates for all possible events
            rates = []
            events = []
            
            # Recovery events - rate 1 for each infected node
            for node in range(self.n):
                if self.node_states[node] == 1:  # If infected
                    rates.append(1)
                    events.append(("recovery", node))
            
            # Transmission events - only attempted along open edges
            for i, edge in enumerate(self.edges):
                if self.edge_states[i] == 1:  # If edge is open
                    node1, node2 = edge
                    # Transmission from node1 to node2
                    if self.node_states[node1] == 1 and self.node_states[node2] == 0:
                        rates.append(self.lambda_val)
                        events.append(("transmit", node1, node2))
                    # Transmission from node2 to node1
                    if self.node_states[node2] == 1 and self.node_states[node1] == 0:
                        rates.append(self.lambda_val)
                        events.append(("transmit", node2, node1))
            
            # Edge update events - opening with rate p*nu, closing with rate (1-p)*nu
            for i, edge in enumerate(self.edges):
                # Edge opening
                if self.edge_states[i] == 0:
                    rates.append(self.p * self.nu)
                    events.append(("open", i))
                # Edge closing
                else:
                    rates.append((1 - self.p) * self.nu)
                    events.append(("close", i))
            
            # If no events possible, end simulation
            if not rates:
                break
            
            # Sample time until next event (exponential distribution)
            total_rate = sum(rates)
            dt = np.random.exponential(1/total_rate) if total_rate > 0 else self.max_time
            next_time = current_time + dt
            
            # If next event would exceed max_time, end simulation
            if next_time > self.max_time:
                break
            
            # Sample which event occurs
            event_idx = np.random.choice(len(events), p=np.array(rates)/total_rate)
            event = events[event_idx]
            
            # Execute the event
            if event[0] == "recovery":
                node = event[1]
                self.node_states[node] = 0
            elif event[0] == "transmit":
                source, target = event[1], event[2]
                self.node_states[target] = 1
            elif event[0] == "open":
                edge_idx = event[1]
                self.edge_states[edge_idx] = 1
            elif event[0] == "close":
                edge_idx = event[1]
                self.edge_states[edge_idx] = 0
            
            # Update time and record state
            current_time = next_time
            self.time_events.append(current_time)
            self.node_states_history.append(self.node_states.copy())
            self.edge_states_history.append(self.edge_states.copy())
    
    def visualize_simulation(self, fps=10, save_path=None):
        """
        Visualize the simulation as an animation.
        
        Args:
            fps (int): Frames per second for the animation
            save_path (str, optional): Path to save the animation as an MP4 file
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a graph for visualization
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        
        # Initialize node colors
        node_colors = ['blue' if state == 0 else 'red' for state in self.node_states_history[0]]
        
        # Calculate positions (fixed throughout the animation)
        if self.n <= 20:  # For small graphs, circular layout works well
            pos = nx.circular_layout(G)
        else:  # For larger graphs, spring layout looks better
            pos = nx.spring_layout(G, seed=42)
        
        # Function to update the animation at each frame
        def update(frame):
            ax.clear()
            
            # Get current edge and node states
            current_edge_states = self.edge_states_history[frame]
            current_node_states = self.node_states_history[frame]
            
            # Update the graph with current open edges
            G.clear()
            G.add_nodes_from(self.nodes)
            edges_to_add = [self.edges[i] for i, state in enumerate(current_edge_states) if state == 1]
            G.add_edges_from(edges_to_add)
            
            # Update node colors based on infection state
            node_colors = ['skyblue' if state == 0 else 'red' for state in current_node_states]
            
            # Draw the graph
            nx.draw(G, pos=pos, with_labels=True, node_color=node_colors, node_size=500, 
                    font_weight='bold', ax=ax, edge_color='gray', width=2.0)
            
            ax.set_title(f'Time: {self.time_events[frame]:.2f}')
            return ax,
        
        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=len(self.time_events), 
                                      interval=1000/fps, blit=False)
        
        # Save the animation if a path is provided
        if save_path:
            ani.save(save_path, writer='ffmpeg', fps=fps)
        
        plt.tight_layout()
        plt.show()
        
    def plot_infection_over_time(self):
        """Plot the number of infected nodes over time."""
        infected_counts = [np.sum(states) for states in self.node_states_history]
        
        plt.figure(figsize=(10, 6))
        plt.step(self.time_events, infected_counts, where='post', color='red', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Number of Infected Nodes')
        plt.title('Infection Count Over Time')
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def visualize_spacetime(self, figsize=(10, 8)):
        """
        Create a space-time diagram showing the infection status of each node over time.
        """
        # Convert node states history to a 2D array
        state_array = np.array(self.node_states_history)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a custom colormap: blue for healthy (0), red for infected (1)
        cmap = ListedColormap(['skyblue', 'red'])
        
        # Plot the space-time diagram
        im = ax.imshow(state_array, aspect='auto', cmap=cmap, interpolation='nearest',
                      extent=[0, self.n, self.time_events[-1], 0])
        
        # Add colorbar and labels
        cbar = plt.colorbar(im, ticks=[0.25, 0.75])
        cbar.ax.set_yticklabels(['Healthy', 'Infected'])
        
        ax.set_xlabel('Node')
        ax.set_ylabel('Time')
        ax.set_title('Space-Time Diagram of Infection Status')
        
        # Set integer ticks for node indices
        ax.set_xticks(np.arange(0.5, self.n, 1))
        ax.set_xticklabels(np.arange(0, self.n))
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Parameters
    n = 15  # number of nodes
    p = 0.3  # probability parameter for edge dynamics
    lambda_val = 3  # infection rate
    nu = 1.0  # rate for edge dynamics
    max_time = 20.0  # simulation time
    
    # Create and run simulation
    simulation = DynamicContactProcess(n=n, p=p, lambda_val=lambda_val, nu=nu, 
                                     max_time=max_time, initial_infected=[0])
    simulation.simulate()
    
    # Visualize the results
    print(f"Simulation completed with {len(simulation.time_events)} events")
    print(f"Final number of infected nodes: {np.sum(simulation.node_states_history[-1])}")
    
    # Generate different visualizations
    simulation.visualize_simulation(fps=5, save_path=None)
    simulation.plot_infection_over_time()
    simulation.visualize_spacetime() 
