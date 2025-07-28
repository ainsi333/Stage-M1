import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

# --- Paramètres de la simulation ---
L = 50  # Taille de la grille (LxL)
p = 0.49  # Probabilité d'ouverture (0 < p < 1)
K = 100   # Nombre d'étapes à simuler

def canon(e):
    """Trie les extrémités d'une arête dans l'ordre lexicographique."""
    return tuple(sorted(e))

# --- Construction du graphe Z^2 ---
def create_grid_graph(L):
    """Crée le graphe de grille de taille LxL de manière optimisée."""
    G = nx.grid_2d_graph(L, L)
    edges = list(G.edges())
    # On s'assure que chaque arête est dans l'ordre canonique (u < v)
    edges = [tuple(sorted(e)) for e in edges]
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    idx_to_edge = {i: e for i, e in enumerate(edges)}
    return G, edges, edge_to_idx, idx_to_edge

# --- Choix de W0 (composante connexe finie contenant (0,0)) ---
def get_initial_W0(G, edge_to_idx, L):
    """Crée la composante initiale W0 de manière optimisée."""
    center = (L//2, L//2)
    W0_nodes = [(center[0]+i, center[1]+j) for i in range(-1,2) for j in range(-1,2)
                if 0 <= center[0]+i < L and 0 <= center[1]+j < L]
    W0_edges = set()
    for u in W0_nodes:
        for v in G.neighbors(u):
            if v in W0_nodes:
                e = tuple(sorted((u, v)))
                if e in edge_to_idx:
                    W0_edges.add(e)
    return W0_edges

# --- Animation et contrôle ---
class PercolationSim:
    def __init__(self, G, edge_status, W, p, K):
        self.G = G
        self.edge_status = edge_status.copy()
        self.p = p
        self.K = K
        self.L = L
        self.edges = list(G.edges())
        self.edges = [tuple(sorted(e)) for e in self.edges]
        self.edge_to_idx = {e: i for i, e in enumerate(self.edges)}
        self.idx_to_edge = {i: e for i, e in enumerate(self.edges)}
        self.W = set(W)
        self.W_history = [set(W)]
        self.status_history = [edge_status.copy()]
        self.step = 0
        self.finished = False
        self.fig, self.ax = plt.subplots(figsize=(7,7))
        plt.subplots_adjust(bottom=0.15)
        self.draw_graph()

    def get_neighbors(self, W):
        """Renvoie les arêtes fermées ayant une extrémité dans W."""
        W_edges = set(W)
        W_nodes = set()
        for e in W:
            W_nodes.update(e)
        neighbor_edges = set()
        for node in W_nodes:
            for v in self.G.neighbors(node):
                e = tuple(sorted((node, v)))
                # Vérifier que l'arête existe dans le graphe et n'est pas déjà dans W
                if e in self.edge_to_idx and e not in W_edges:
                    # Vérifier que l'arête est fermée
                    if self.edge_status[self.edge_to_idx[e]] == 0:
                        neighbor_edges.add(e)
        return neighbor_edges

    def get_random_connected_component(self, edge, forbidden_edges, max_size=100):
        """Génère une composante connexe à partir d'une arête en suivant la logique :
        1. On part de e_new (l'arête tirée à la frontière)
        2. On regarde ses voisins qui ne touchent pas W
        3. On ouvre chaque voisin avec proba p
        4. Pour chaque voisin ouvert, on répète le procédé
        5. On s'arrête quand tous les voisins sont fermés"""
        if edge not in self.edge_to_idx:
            return {edge}
        
        comp = {edge}  # La composante connexe
        frontier = {edge}  # Les arêtes dont on va explorer les voisins
        
        while frontier and len(comp) < max_size:
            current = frontier.pop()
            new_frontier = set()
            
            # Pour chaque nœud de l'arête courante
            for node in current:
                # On regarde toutes les arêtes voisines
                for v in self.G.neighbors(node):
                    e2 = tuple(sorted((node, v)))
                    
                    # Vérifier que l'arête est valide
                    if (e2 in self.edge_to_idx and  # L'arête existe dans le graphe
                        e2 not in comp and          # Pas déjà dans la composante
                        self.edge_status[self.edge_to_idx[e2]] == 0 and  # Arête fermée
                        e2 not in forbidden_edges):  # Pas dans les arêtes interdites
                        
                        # Vérifier que l'arête ne touche pas W sauf par edge
                        touches_W = False
                        for w in e2:
                            for w_neighbor in self.G.neighbors(w):
                                e3 = tuple(sorted((w, w_neighbor)))
                                if e3 in forbidden_edges and e3 != edge:
                                    touches_W = True
                                    break
                            if touches_W:
                                break
                        
                        if not touches_W:
                            # On tire avec probabilité p si on ouvre cette arête
                            if random.random() < self.p:
                                comp.add(e2)
                                new_frontier.add(e2)
            
            # Si on n'a pas trouvé de nouveaux voisins à ouvrir, on s'arrête
            if not new_frontier:
                break
            
            # Sinon, on continue avec les nouveaux voisins ouverts
            frontier.update(new_frontier)
        
        return comp

    def next_step(self):
        """Exécute une étape du processus."""
        if self.finished or self.step >= self.K:
            return False
            
        W = set(self.W)
        if len(W) == 0:
            self.finished = True
            return False
            
        neighbor_edges = self.get_neighbors(W)
        N = len(neighbor_edges)
        M = len(W)
        
        if M == 0 and N == 0:
            self.finished = True
            return False
            
        rate = M*self.p + N*(1-self.p)
        if rate == 0:
            self.finished = True
            return False
            
        ouverture_prob = (M*self.p) / rate
        
        if N > 0 and random.random() < ouverture_prob:
            # OUVERTURE
            e_new = random.choice(list(neighbor_edges))
            forbidden = W
            comp = self.get_random_connected_component(e_new, forbidden, max_size=100)
            for e in comp:
                if e in self.edge_to_idx:  # Vérifier que l'arête existe
                    self.edge_status[self.edge_to_idx[e]] = 1
            W_new = W.union(comp)
        elif M > 0:
            # FERMETURE
            e_close = random.choice(list(W))
            if e_close in self.edge_to_idx:  # Vérifier que l'arête existe
                self.edge_status[self.edge_to_idx[e_close]] = 0
            W_new = W - {e_close}
        else:
            self.finished = True
            return False
            
        self.W = set(W_new)
        self.W_history.append(set(W_new))
        self.status_history.append(self.edge_status.copy())
        self.step += 1
        self.draw_graph()
        return True

    def run(self):
        """Exécute la simulation complète."""
        while self.next_step():
            plt.pause(0.5)

    def draw_graph(self):
        """Dessine l'état actuel du graphe."""
        self.ax.clear()
        pos = {(i, j): (i, j) for i in range(self.L) for j in range(self.L)}
        open_edges = [self.idx_to_edge[i] for i, s in enumerate(self.edge_status) if s == 1]
        closed_edges = [self.idx_to_edge[i] for i, s in enumerate(self.edge_status) if s == 0]
        
        # Dessiner les arêtes fermées en gris clair
        nx.draw_networkx_edges(self.G, pos, edgelist=closed_edges, 
                             style='solid', alpha=0.1, edge_color='gray', 
                             width=0.5, ax=self.ax)
        
        # Dessiner les arêtes ouvertes en noir
        nx.draw_networkx_edges(self.G, pos, edgelist=open_edges, 
                             style='solid', width=1.5, edge_color='black', 
                             ax=self.ax)
        
        # Dessiner les nœuds en très petit et en gris clair
        nx.draw_networkx_nodes(self.G, pos, node_size=2, 
                             node_color='lightgray', alpha=0.5, ax=self.ax)
        
        self.ax.set_title(f'Étape {self.step}/{self.K}')
        self.ax.set_axis_off()
        self.fig.canvas.draw_idle()

# --- Lancer la simulation ---
if __name__ == "__main__":
    G, edges, edge_to_idx, idx_to_edge = create_grid_graph(L)
    edge_status = np.zeros(len(edges), dtype=int)
    W = get_initial_W0(G, edge_to_idx, L)
    for e in W:
        edge_status[edge_to_idx[e]] = 1
    
    sim = PercolationSim(G, edge_status, W, p, K)
    sim.run()
    plt.show() 
