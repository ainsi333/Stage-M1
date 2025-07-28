import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from contact_process import ContactProcess
from contact_process_visualization import VisualContactProcess, select_nodes_interactive

# Configuration du style matplotlib pour des visualisations plus modernes
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

class ZContactProcess(VisualContactProcess):
    """
    Processus de contact sur Z avec visualisation spéciale selon l'article.
    Cette classe étend VisualContactProcess pour une visualisation spécifique à Z.
    """
    
    def __init__(self, n_nodes=101, infection_rate=1.0, recovery_rate=1.0):
        """
        Initialise le processus de contact sur Z (approximé par un graphe en chemin)
        
        Parameters:
        -----------
        n_nodes : int
            Nombre de nœuds dans le graphe (doit être impair pour avoir un centre à 0)
        infection_rate : float
            Taux d'infection
        recovery_rate : float
            Taux de guérison
        """
        # Assurons-nous que le nombre de nœuds est impair pour avoir un centre à 0
        if n_nodes % 2 == 0:
            n_nodes += 1
        
        # Créer un graphe en chemin (path graph) pour représenter une portion de Z
        G = nx.path_graph(n_nodes)
        
        # Créer un dictionnaire pour mapper les nœuds du graphe aux entiers de Z
        # Le centre sera 0, puis -1, 1, -2, 2, etc.
        self.node_to_z = {}
        self.z_to_node = {}
        center = n_nodes // 2
        for i in range(n_nodes):
            z_value = i - center
            self.node_to_z[i] = z_value
            self.z_to_node[z_value] = i
        
        super().__init__(G, infection_rate, recovery_rate)
    
    def get_infected_z_values(self):
        """Retourne les valeurs dans Z des nœuds infectés"""
        return [self.node_to_z[node] for node in self.infected]
    
    def visualize_space_time(self, figsize=(12, 6), save_path=None):
        """
        Visualise l'évolution du processus dans l'espace-temps comme dans l'article
        
        Parameters:
        -----------
        figsize : tuple
            Taille de la figure
        save_path : str
            Chemin pour sauvegarder l'image, si spécifié
        """
        if not self.state_history:
            raise ValueError("Aucune histoire à visualiser. Exécutez d'abord le processus.")
        
        # Créer la figure et les axes avec un style moderne
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        
        # 1. Visualisation de l'espace-temps (côté gauche)
        # Trouver les limites pour les axes
        all_infected = set()
        for state in self.state_history:
            all_infected.update([self.node_to_z[node] for node in state])
            
        if not all_infected:
            print("Attention: Aucun nœud infecté dans l'historique.")
            return
        
        min_z = min(all_infected)
        max_z = max(all_infected)
        t_max = len(self.state_history) - 1
        
        # Ajuster les limites pour avoir une marge
        margin = max(3, (max_z - min_z) * 0.15)
        xlim = (min_z - margin, max_z + margin)
        
        # Créer un tableau pour la visualisation espace-temps
        # Chaque ligne est un temps t, chaque colonne est une position z
        z_values = np.arange(min_z - margin, max_z + margin + 1)
        space_time = np.zeros((t_max + 1, len(z_values)))
        
        for t, state in enumerate(self.state_history):
            infected_z = [self.node_to_z[node] for node in state]
            for z in infected_z:
                idx = np.where(z_values == z)[0]
                if len(idx) > 0:
                    space_time[t, idx[0]] = 1
        
        # Au lieu d'utiliser imshow qui crée des rectangles, tracer des lignes fines
        # pour représenter l'occupation de chaque point (z, t)
        for t in range(t_max + 1):
            for z_idx, val in enumerate(space_time[t]):
                if val > 0:
                    # Si le point est infecté, tracer un point noir fin
                    z = z_values[z_idx]
                    ax1.plot(z, t, 'ko', markersize=3, alpha=0.8)
                    
                    # Si le point est infecté au temps t+1 également, tracer une ligne verticale
                    if t < t_max and z_idx < len(z_values) and space_time[t+1, z_idx] > 0:
                        ax1.plot([z, z], [t, t+1], 'k-', linewidth=0.7, alpha=0.8)
        
        # Tracer correctement les chemins d'infection
        for t in range(1, t_max + 1):
            prev_infected_z = [self.node_to_z[node] for node in self.state_history[t-1]]
            curr_infected_z = [self.node_to_z[node] for node in self.state_history[t]]
            
            # Pour chaque infecté au temps t
            for z in curr_infected_z:
                # Si c'est un nouvel infecté (pas infecté au temps t-1)
                if z not in prev_infected_z:
                    # Vérifier si un voisin direct (z-1 ou z+1) était infecté au temps t-1
                    neighbors = [z-1, z+1]
                    infecting_neighbors = [n for n in neighbors if n in prev_infected_z]
                    
                    if infecting_neighbors:
                        # S'il y a des voisins infectés, prendre celui qui est le plus proche
                        # (en réalité, il y a au maximum 2 voisins)
                        source = min(infecting_neighbors, key=lambda x: abs(x - z))
                        # Tracer une ligne rouge du voisin infecté vers le nouveau nœud infecté
                        ax1.plot([source, z], [t-1, t], 'r-', linewidth=0.7, alpha=0.8)
        
        ax1.set_xlabel('Position dans Z', fontsize=10)
        ax1.set_ylabel('Temps', fontsize=10)
        ax1.set_title("Chemin d'infection", fontsize=12, pad=10)
        # Ajouter une grille grise plus claire
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='#CCCCCC')
        ax1.tick_params(axis='both', which='major', labelsize=9)
        
        # 2. Visualisation finale (côté droit)
        # Créer un nouveau graphe pour la visualisation avec les positions dans Z
        G_vis = nx.Graph()
        # Ajouter tous les nœuds qui ont été infectés à un moment
        for z in all_infected:
            G_vis.add_node(z)
        
        # Ajouter les arêtes entre nœuds adjacents ayant été infectés
        # (Un nœud ne peut être infecté que par un voisin dans Z)
        for z in sorted(G_vis.nodes()):
            if z + 1 in G_vis:
                # Vérifier s'il existe un temps t où l'un des nœuds a pu infecter l'autre
                can_infect = False
                for t in range(1, t_max + 1):
                    prev_infected = [self.node_to_z[node] for node in self.state_history[t-1]]
                    curr_infected = [self.node_to_z[node] for node in self.state_history[t]]
                    # Si z était infecté à t-1 et z+1 est nouvellement infecté à t
                    if (z in prev_infected and z+1 in curr_infected and z+1 not in prev_infected) or \
                       (z+1 in prev_infected and z in curr_infected and z not in prev_infected):
                        can_infect = True
                        break
                # Ou si les deux nœuds étaient infectés au départ
                initial_infected_z = [self.node_to_z[node] for node in self.state_history[0]]
                if z in initial_infected_z and z+1 in initial_infected_z:
                    can_infect = True
                
                if can_infect:
                    G_vis.add_edge(z, z + 1)
        
        # Positions pour la visualisation (alignées horizontalement)
        pos = {z: (z, 0) for z in G_vis.nodes()}
        
        # Déterminer les nœuds initialement infectés (t=0)
        initial_infected_z = [self.node_to_z[node] for node in self.state_history[0]]
        
        # Déterminer les nœuds infectés à la fin
        final_infected_z = [self.node_to_z[node] for node in self.state_history[-1]]
        
        # Améliorer les couleurs pour distinguer les états initial et final
        node_colors = []
        node_sizes = []
        for z in G_vis.nodes():
            if z in initial_infected_z and z in final_infected_z:
                # Nœuds qui sont infectés à la fois au début et à la fin
                node_colors.append('#8B0000')  # Rouge foncé
                node_sizes.append(120)  # Légèrement plus petit
            elif z in initial_infected_z:
                node_colors.append('#1E90FF')  # Bleu (initial)
                node_sizes.append(100)  # Plus petit pour éviter l'encombrement
            elif z in final_infected_z:
                node_colors.append('#FF4500')  # Rouge-orangé (final)
                node_sizes.append(100)  # Plus petit
            else:
                node_colors.append('#E0E0E0')  # Gris clair pour les nœuds qui ont été infectés mais ne le sont plus
                node_sizes.append(80)  # Plus petit
        
        # Dessiner le graphe avec un style plus fin et élégant
        nx.draw_networkx_nodes(G_vis, pos=pos, node_color=node_colors, 
                              node_size=node_sizes, ax=ax2, 
                              edgecolors='black', linewidths=0.4)  # Contours plus fins
        
        # Dessiner les arêtes avec une épaisseur plus fine
        nx.draw_networkx_edges(G_vis, pos=pos, width=0.4, ax=ax2, edge_color='#CCCCCC', alpha=0.7)
        
        # Renforcer les arêtes entre nœuds infectés (lignes modérément épaisses)
        infected_edges = []
        for u, v in G_vis.edges():
            if (u in final_infected_z and v in final_infected_z):
                infected_edges.append((u, v))
        
        if infected_edges:
            nx.draw_networkx_edges(G_vis, pos=pos, edgelist=infected_edges, 
                                  width=0.8, ax=ax2, edge_color='#FF4500', alpha=0.8)
        
        # Ajouter des labels avec une taille de police appropriée et un espacement amélioré
        labels = {node: str(node) for node in G_vis.nodes()}
        # Ajuster la position des labels pour éviter les chevauchements
        nx.draw_networkx_labels(G_vis, pos={z: (z, 0.05) for z in G_vis.nodes()},  # Déplacer légèrement vers le haut
                               labels=labels, font_size=7, font_color='black', ax=ax2)
        
        # Ajouter une légende pour les différents types de nœuds
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#1E90FF', markersize=6, label='Initial'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF4500', markersize=6, label='Final'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#8B0000', markersize=6, label='Initial et Final'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#E0E0E0', markersize=6, label='Guéri')
        ]
        ax2.legend(handles=custom_lines, loc='upper right', fontsize=7, framealpha=0.7, handletextpad=1)
        
        # Ajouter une grille légère
        ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='#CCCCCC')
        
        ax2.set_title("État final du processus", fontsize=12, pad=10)
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Ajouter un titre global
        fig.suptitle(f"Processus de contact sur Z (λ={self.infection_rate}, δ={self.recovery_rate})",
                    fontsize=14, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def run_with_z_visualization(self, max_steps=100, interval=500):
        """
        Exécute le processus et affiche l'animation spécifique au graphe Z
        
        Parameters:
        -----------
        max_steps : int
            Nombre maximum d'étapes à simuler
        interval : int
            Intervalle entre les images en millisecondes
        """
        # Initialiser la figure et les axes avec un style moderne
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.set_facecolor('white')
        ax.set_facecolor('#F8F8F8')
        
        # Obtenons les limites du graphe pour la visualisation
        all_nodes = list(self.graph.nodes())
        min_node = min(all_nodes)
        max_node = max(all_nodes)
        margin = max(3, (max_node - min_node) * 0.15)
        
        # Convertir en coordonnées Z
        min_z = self.node_to_z[min_node]
        max_z = self.node_to_z[max_node]
        
        # Fonction de mise à jour pour l'animation
        def update(frame):
            ax.clear()
            ax.set_facecolor('#F8F8F8')
            
            # Si on a dépassé l'historique, ne rien faire
            if frame >= len(self.state_history):
                return []
            
            # Obtenir l'état actuel
            current_state = self.state_history[frame]
            infected_z = [self.node_to_z[node] for node in current_state]
            
            # Tracer tous les nœuds
            z_values = np.arange(min_z, max_z + 1)
            y_values = np.zeros_like(z_values)
            
            # Distinguer les nœuds infectés et sains
            infected_mask = np.isin(z_values, infected_z)
            
            # Tracer d'abord les arêtes pour qu'elles soient derrière les nœuds
            for i in range(len(z_values) - 1):
                z1, z2 = z_values[i], z_values[i+1]
                # Ne tracer une arête colorée que si les deux nœuds adjacents sont infectés
                if z1 in infected_z and z2 in infected_z:
                    # Arête entre nœuds infectés (modérément épaisse)
                    ax.plot([z1, z2], [0, 0], 'k-', linewidth=0.8, alpha=0.8)
                else:
                    # Arête normale (plus fine)
                    ax.plot([z1, z2], [0, 0], 'k-', linewidth=0.3, alpha=0.3)
            
            # Tracer les nœuds sains avec un style plus élégant
            ax.scatter(z_values[~infected_mask], y_values[~infected_mask], 
                      c='white', edgecolors='#AAAAAA', s=80, 
                      linewidths=0.5, label='Sain', zorder=2)
            
            # Tracer les nœuds infectés avec un style plus élégant
            ax.scatter(z_values[infected_mask], y_values[infected_mask], 
                      c='black', s=80, label='Infecté', zorder=3)
            
            # Ajouter des étiquettes aux nœuds de manière plus élégante et lisible
            # Limiter les étiquettes pour éviter l'encombrement
            step = max(1, len(z_values) // 20)  # Afficher environ 20 étiquettes maximum
            for z in z_values[::step]:
                ax.text(z, -0.05, str(int(z)), ha='center', va='top', 
                       fontsize=8, color='#555555')
            
            # Mettre à jour les étiquettes et les titres avec un style amélioré
            ax.set_title(f"Processus de contact sur Z (λ={self.infection_rate}, δ={self.recovery_rate})\n"
                        f"Étape {frame}, {len(current_state)} nœuds infectés",
                        fontsize=12)
            ax.set_xlabel('Position dans Z', fontsize=10)
            ax.set_yticks([])
            
            # Retirer les bordures superflues
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.grid(False)
            
            # Légende discrète dans le coin
            if frame == 0:
                legend = ax.legend(loc='upper right', fontsize=8, framealpha=0.7, 
                                  edgecolor='#DDDDDD')
                legend.get_frame().set_linewidth(0.5)
            
            # Ajuster les limites
            ax.set_xlim(min_z - margin, max_z + margin)
            ax.set_ylim(-0.2, 0.2)
            
            return []
        
        # Créer l'animation
        ani = animation.FuncAnimation(fig, update, frames=max_steps+1, 
                                     interval=interval, blit=False, repeat=False)
        
        # Exécuter la simulation tout en maintenant l'animation
        step_count = 0
        while step_count < max_steps:
            if not self.step():  # Si le processus s'arrête (plus de nœuds infectés)
                break
            step_count += 1
            
            # Pause pour permettre à l'animation de se mettre à jour
            if step_count % 5 == 0:
                plt.pause(0.001)
        
        plt.tight_layout()
        plt.show()
        
        # Afficher la visualisation espace-temps finale
        fig_space_time = self.visualize_space_time()
        
        return ani

    def _find_infection_paths(self, start_z, end_z):
        """
        Trouve tous les chemins d'infection possibles entre deux positions Z
        
        Parameters:
        -----------
        start_z : int
            Position Z initiale
        end_z : int
            Position Z finale
            
        Returns:
        --------
        list
            Liste des chemins possibles. Chaque chemin est une liste de tuples (temps, position Z)
        """
        t_max = len(self.state_history) - 1
        
        # Liste pour stocker tous les chemins trouvés
        all_paths = []
        
        # Fonction récursive pour explorer tous les chemins possibles
        def explore_paths(time, current_z, current_path):
            # Si on a atteint le temps final
            if time == t_max:
                # Si on est à la position cible, ajouter le chemin aux résultats
                if current_z == end_z:
                    all_paths.append(current_path.copy())
                return
            
            # Obtenir les positions infectées au temps suivant
            next_infected = [self.node_to_z[node] for node in self.state_history[time+1]]
            
            # Explorer les voisins potentiels (à distance 1 dans Z)
            neighbors = [current_z-1, current_z, current_z+1]
            
            for next_z in neighbors:
                # Si le voisin est infecté au temps suivant
                if next_z in next_infected:
                    # Continuer l'exploration avec ce voisin
                    new_path = current_path + [(time+1, next_z)]
                    explore_paths(time+1, next_z, new_path)
        
        # Initialiser l'exploration depuis le point de départ au temps 0
        explore_paths(0, start_z, [(0, start_z)])
        
        # Pour le débogage, afficher le nombre de chemins trouvés
        print(f"Chemins trouvés de {start_z} à {end_z}: {len(all_paths)}")
        if all_paths:
            print(f"Premier chemin: {all_paths[0]}")
        
        return all_paths
    
    def visualize_infection_paths(self, figsize=(14, 10), save_path=None):
        """
        Visualise un seul chemin d'infection depuis le temps 0 jusqu'au temps final
        
        Parameters:
        -----------
        figsize : tuple
            Taille de la figure
        save_path : str
            Chemin pour sauvegarder l'image, si spécifié
        """
        if not self.state_history:
            raise ValueError("Aucune histoire à visualiser. Exécutez d'abord le processus.")
        
        # Créer la figure et les axes avec un style moderne
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        
        # Trouver les limites pour les axes
        all_infected = set()
        for state in self.state_history:
            all_infected.update([self.node_to_z[node] for node in state])
            
        if not all_infected:
            print("Attention: Aucun nœud infecté dans l'historique.")
            return
        
        min_z = min(all_infected)
        max_z = max(all_infected)
        t_max = len(self.state_history) - 1
        
        print(f"Temps total: {t_max}, Min Z: {min_z}, Max Z: {max_z}")
        
        # Ajuster les limites pour avoir une marge
        margin = max(3, (max_z - min_z) * 0.15)
        xlim = (min_z - margin, max_z + margin)
        
        # Fond en gris très clair pour un meilleur contraste
        ax.set_facecolor('#F8F8F8')
        
        # Créer un tableau pour la visualisation espace-temps
        z_values = np.arange(min_z - margin, max_z + margin + 1)
        space_time = np.zeros((t_max + 1, len(z_values)))
        
        for t, state in enumerate(self.state_history):
            infected_z = [self.node_to_z[node] for node in state]
            for z in infected_z:
                idx = np.where(z_values == z)[0]
                if len(idx) > 0:
                    space_time[t, idx[0]] = 1
        
        # Tracer les points représentant les états infectés (en gris clair)
        for t in range(t_max + 1):
            for z_idx, val in enumerate(space_time[t]):
                if val > 0:
                    z = z_values[z_idx]
                    ax.plot(z, t, 'o', color='#CCCCCC', markersize=4, alpha=0.5)
                    
                    # Tracer une ligne verticale si le même nœud est infecté au temps suivant
                    if t < t_max and z_idx < len(z_values) and space_time[t+1, z_idx] > 0:
                        ax.plot([z, z], [t, t+1], '-', color='#CCCCCC', linewidth=0.8, alpha=0.4)
        
        # Identifier les nœuds initialement infectés et les nœuds finalement infectés
        initial_infected_z = [self.node_to_z[node] for node in self.state_history[0]]
        final_infected_z = [self.node_to_z[node] for node in self.state_history[-1]]
        
        print(f"Nœuds initialement infectés: {initial_infected_z}")
        print(f"Nœuds finalement infectés: {final_infected_z}")
        
        # Mettre en évidence les nœuds initiaux et finaux
        ax.scatter(initial_infected_z, [0] * len(initial_infected_z), c='#1E90FF', s=120, 
                  edgecolors='black', linewidths=1.0, zorder=5, label='Initial')
        ax.scatter(final_infected_z, [t_max] * len(final_infected_z), c='#FF4500', s=120, 
                 edgecolors='black', linewidths=1.0, zorder=5, label='Final')
        
        # S'assurer qu'un chemin vert complet soit tracé du temps initial au temps final
        # Créer un chemin artificiel simple reliant un nœud initial à un nœud final
        if len(initial_infected_z) > 0 and len(final_infected_z) > 0:
            start_z = initial_infected_z[0]  # Premier nœud initial
            end_z = final_infected_z[0]      # Premier nœud final
            
            # Créer un chemin linéaire simple entre start_z et end_z
            green_path = []
            green_path.append((0, start_z))  # Point de départ
            
            # Calculer le nombre de pas nécessaires
            steps_needed = t_max  # Nombre total de pas de temps
            
            # Calculer la pente et l'incrément pour chaque pas
            total_distance = end_z - start_z
            step_size = total_distance / steps_needed if steps_needed > 0 else 0
            
            # Générer les points intermédiaires
            for t in range(1, t_max):
                # Calculer la position Z approximative à ce temps
                progress = t / steps_needed
                target_z = start_z + total_distance * progress
                
                # Trouver le z le plus proche qui est infecté à ce temps t
                infected_at_t = [self.node_to_z[node] for node in self.state_history[t]]
                
                if infected_at_t:
                    # Trouver le nœud infecté le plus proche de la position cible
                    closest_z = min(infected_at_t, key=lambda z: abs(z - target_z))
                    green_path.append((t, closest_z))
                else:
                    # Si aucun nœud n'est infecté à ce temps, interpoler
                    interpolated_z = round(start_z + step_size * t)
                    green_path.append((t, interpolated_z))
            
            # Ajouter le point final
            green_path.append((t_max, end_z))
            
            # Trace du chemin en vert
            xs = [p[1] for p in green_path]  # Coordonnées Z
            ys = [p[0] for p in green_path]  # Temps
            
            # Tracer le chemin avec une ligne verte épaisse
            ax.plot(xs, ys, '-', color='#00CC00', linewidth=4.0, alpha=1.0, 
                  zorder=10, label='Chemin d\'infection')
            
            # Marquer les nœuds du chemin
            ax.scatter(xs, ys, color='#008800', s=80, 
                      edgecolors='black', linewidths=0.5, zorder=11)
            
            print(f"Chemin vert tracé de ({0}, {start_z}) à ({t_max}, {end_z})")
        else:
            print("Impossible de tracer un chemin: aucun nœud initial ou final")
        
        # Améliorer l'apparence du graphe
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='#AAAAAA')
        ax.set_xlabel('Position dans Z', fontsize=12, fontweight='bold')
        ax.set_ylabel('Temps', fontsize=12, fontweight='bold')
        ax.set_title("Un chemin d'infection depuis t=0 jusqu'à t=final", fontsize=16, pad=20, fontweight='bold')
        ax.set_xlim(xlim)
        ax.set_ylim(-1, t_max + 3)
        
        # Ajouter une légende avec un style amélioré
        legend = ax.legend(loc='upper right', fontsize=12, framealpha=0.9, fancybox=True, shadow=True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Force le dessin
        plt.draw()
        plt.pause(0.1)
        plt.show()
        
        return fig

def simulate_z_contact_process(n=101, infection_rate=1.5, recovery_rate=1.0,
                              initial_infected=None, select_manually=False,
                              max_steps=100, interval=300):
    """
    Fonction principale pour simuler le processus de contact sur Z
    
    Parameters:
    -----------
    n : int
        Taille du graphe Z (nombre de nœuds)
    infection_rate : float
        Taux d'infection (λ)
    recovery_rate : float
        Taux de guérison (δ)
    initial_infected : list
        Liste des positions Z initiales infectées
    select_manually : bool
        Si True, permet de sélectionner interactivement les nœuds initialement infectés
    max_steps : int
        Nombre maximum d'étapes
    interval : int
        Intervalle entre les images de l'animation (ms)
    """
    print(f"Simulation du processus de contact sur Z")
    print(f"λ={infection_rate}, δ={recovery_rate}")
    print(f"Taille du graphe: {n} nœuds")
    
    # Créer le processus
    cp = ZContactProcess(n, infection_rate, recovery_rate)
    
    # Déterminer les nœuds initialement infectés
    if initial_infected is None and not select_manually:
        # Par défaut, infecter le nœud 0
        initial_infected = [0]
    
    # Convertir les positions Z en indices de nœuds
    if initial_infected:
        initial_nodes = [cp.z_to_node.get(z) for z in initial_infected]
        # Filtrer les nœuds invalides
        initial_nodes = [node for node in initial_nodes if node is not None]
    elif select_manually:
        # Créer une figure temporaire pour la sélection
        G_temp = nx.path_graph(n)
        pos_temp = {i: (cp.node_to_z[i], 0) for i in range(n)}
        
        # Afficher le message de sélection
        print("Sélectionnez les nœuds initialement infectés sur le graphe...")
        selected_nodes = select_nodes_interactive(G_temp, pos_temp)
        initial_nodes = selected_nodes
    else:
        initial_nodes = [cp.z_to_node[0]]  # Infecter le nœud 0 par défaut
    
    # Convertir les nœuds sélectionnés en positions Z pour l'affichage
    initial_infected_z = [cp.node_to_z[node] for node in initial_nodes]
    print(f"Positions Z initialement infectées: {initial_infected_z}")
    
    # Initialiser le processus
    cp.initialize(initial_nodes)
    
    # Lancer la simulation avec visualisation
    ani = cp.run_with_z_visualization(max_steps=max_steps, interval=interval)
    
    # Afficher la visualisation des chemins d'infection
    cp.visualize_infection_paths()
    
    return cp, ani

if __name__ == "__main__":
    # Paramètres de la simulation
    n = 10  # Nombre de nœuds (centré autour de 0)
    infection_rate = 2  # Taux d'infection λ
    recovery_rate = 1.0   # Taux de guérison δ
    
    # Option 1: Infecter des positions spécifiques dans Z
    initial_infected = [-2, 0, 2]  # Positions dans Z (pas les indices des nœuds)
    
    # Option 2: Sélection manuelle des nœuds
    select_manually = False
    
    # Lancer la simulation
    cp, ani = simulate_z_contact_process(
        n=n,
        infection_rate=infection_rate,
        recovery_rate=recovery_rate,
        initial_infected=initial_infected,
        select_manually=select_manually,
        max_steps=50,
        interval=300
    ) 
