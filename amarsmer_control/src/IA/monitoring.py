#!/usr/bin/env python3

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import math
import time
from collections import deque
import os
import queue

class RobotMonitor:
    """
    Module de monitoring pour visualiser en temps réel les différentes métriques 
    de l'apprentissage en ligne du robot Pioneer.
    """
    
    def __init__(self, max_points=None, update_interval=300, world_bounds=(-10, 10, -10, 10)):
        """
        Initialise le moniteur de robot.
        
        Args:
            max_points (int): Nombre maximum de points à conserver dans l'historique
            update_interval (int): Intervalle de mise à jour de l'affichage en ms
            world_bounds (tuple): Limites du monde (x_min, x_max, y_min, y_max)
        """
        # Configuration
        self.max_points = max_points
        self.update_interval = update_interval
        self.world_bounds = world_bounds  # Nouvelle propriété pour définir les limites du monde
        
        # Initialize data queue
        self.data_queue = queue.Queue()
        
        # Données à suivre - now using max_points
        self.cost_history = deque(maxlen=max_points)
        self.trajectory = deque(maxlen=max_points)
        self.wheel_speeds = deque(maxlen=max_points)
        self.gradients = deque(maxlen=max_points)
        self.distances = deque(maxlen=max_points)
        self.timestamps = deque(maxlen=max_points)
        self.target = [0, 0, 0]  # Position cible [x, y, theta]
        self.start_set = False  # Flag pour dessiner le point de départ une seule fois

        
        # La figure sera initialisée dans start()
        self.fig = None
        self.ani = None
        self.running = False
        
    def setup_plots(self):
        """Initialise la structure des graphiques"""
        plt.ion()  # Mode interactif pour mise à jour en temps réel
        
        # Créer la figure principale avec GridSpec pour un meilleur contrôle
        self.fig = plt.figure(figsize=(15, 10), facecolor='#f8f9fa')
        self.fig.canvas.manager.set_window_title('Robot Pioneer - Monitoring en temps réel')
        gs = GridSpec(3, 3, figure=self.fig)
        
        # Graphique de la trajectoire (plus grand, occupe 2 lignes)
        self.ax_trajectory = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_trajectory.set_title('Trajectoire du robot', fontweight='bold')
        self.ax_trajectory.set_xlabel('Position X (m)')
        self.ax_trajectory.set_ylabel('Position Y (m)')
        self.ax_trajectory.grid(True, linestyle='--', alpha=0.7)
        
        # Définir immédiatement les limites du graphique de trajectoire selon world_bounds
        self.ax_trajectory.set_xlim(self.world_bounds[0], self.world_bounds[1])
        self.ax_trajectory.set_ylim(self.world_bounds[2], self.world_bounds[3])
        
        # Ajouter des lignes de référence à l'origine
        self.ax_trajectory.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.ax_trajectory.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Éléments de la trajectoire
        self.trajectory_line, = self.ax_trajectory.plot([], [], 'b-', linewidth=2, label='Trajectoire')
        self.trajectory_point, = self.ax_trajectory.plot([], [], 'ro', markersize=8, label='Position actuelle')
        self.target_point, = self.ax_trajectory.plot([], [], 'g*', markersize=12, label='Cible')
        self.start_point, = self.ax_trajectory.plot([], [], 'y^', markersize=10, label='Départ')
        
        
        self.ax_trajectory.legend(loc='upper right')
        
        # Graphique de la fonction de coût
        self.ax_cost = self.fig.add_subplot(gs[0, 2])
        self.ax_cost.set_title('Fonction de coût', fontweight='bold')
        self.ax_cost.set_xlabel('Temps (s)')
        self.ax_cost.set_ylabel('Coût')
        self.ax_cost.grid(True, linestyle='--', alpha=0.7)
        self.cost_line, = self.ax_cost.plot([], [], 'r-', linewidth=2)
        
        # Graphique des vitesses des roues
        self.ax_wheels = self.fig.add_subplot(gs[1, 2])
        self.ax_wheels.set_title('Vitesses des roues', fontweight='bold')
        self.ax_wheels.set_xlabel('Temps (s)')
        self.ax_wheels.set_ylabel('Vitesse (rad/s)')
        self.ax_wheels.grid(True, linestyle='--', alpha=0.7)
        self.left_wheel_line, = self.ax_wheels.plot([], [], 'b-', linewidth=2, label='Roue gauche')
        self.right_wheel_line, = self.ax_wheels.plot([], [], 'g-', linewidth=2, label='Roue droite')
        self.ax_wheels.legend(loc='upper right')
        
        # Graphique du gradient
        self.ax_gradient = self.fig.add_subplot(gs[2, 0])
        self.ax_gradient.set_title('Gradient', fontweight='bold')
        self.ax_gradient.set_xlabel('Temps (s)')
        self.ax_gradient.set_ylabel('Valeur du gradient')
        self.ax_gradient.grid(True, linestyle='--', alpha=0.7)
        self.grad_line_left, = self.ax_gradient.plot([], [], 'c-', linewidth=2, label='Grad roue gauche')
        self.grad_line_right, = self.ax_gradient.plot([], [], 'm-', linewidth=2, label='Grad roue droite')
        self.ax_gradient.legend(loc='upper right')
        
        # Graphiques des distances (erreurs) - deux graphes côte à côte
        # Erreurs X et Y
        self.ax_distance_xy = self.fig.add_subplot(gs[2, 1])
        self.ax_distance_xy.set_title('Erreurs de position X et Y', fontweight='bold')
        self.ax_distance_xy.set_xlabel('Temps (s)')
        self.ax_distance_xy.set_ylabel('Erreur')
        self.ax_distance_xy.grid(True, linestyle='--', alpha=0.7)
        self.distance_x_line, = self.ax_distance_xy.plot([], [], 'r-', linewidth=2, label='Erreur X')
        self.distance_y_line, = self.ax_distance_xy.plot([], [], 'g-', linewidth=2, label='Erreur Y')
        self.ax_distance_xy.legend(loc='upper right')
        
        # Erreur θ
        self.ax_distance_theta = self.fig.add_subplot(gs[2, 2])
        self.ax_distance_theta.set_title('Erreur θ', fontweight='bold')
        self.ax_distance_theta.set_xlabel('Temps (s)')
        self.ax_distance_theta.set_ylabel('Erreur θ')
        self.ax_distance_theta.grid(True, linestyle='--', alpha=0.7)
        self.distance_theta_line, = self.ax_distance_theta.plot([], [], 'b-', linewidth=2, label='Erreur θ')
        self.ax_distance_theta.legend(loc='upper right')
        
        # Ajuster les espacements
        self.fig.tight_layout(pad=3.0)
        
        # Timer pour les mises à jour - Utiliser un nombre explicite de sauvegardes
        self.ani = animation.FuncAnimation(
            self.fig, 
            self.update_plots, 
            interval=self.update_interval,
            blit=False,
            cache_frame_data=False,  # Désactiver le cache des frames
            save_count=100  # Limiter le nombre de frames sauvegardées
        )
    
    def update_plots(self, frame):
        """Mise à jour des graphiques avec les dernières données"""
        # Traiter toutes les données en attente dans la queue
        while not self.data_queue.empty():
            try:
                data_point = self.data_queue.get_nowait()
                if data_point['type'] == 'data':
                    # Ajouter les données aux historiques
                    if 'cost' in data_point:
                        self.cost_history.append(data_point['cost'])
                    if 'position' in data_point:
                        self.trajectory.append(data_point['position'])
                    if 'wheel_speeds' in data_point:
                        self.wheel_speeds.append(data_point['wheel_speeds'])
                    if 'gradient' in data_point:
                        self.gradients.append(data_point['gradient'])
                    if 'distances' in data_point:
                        self.distances.append(data_point['distances'])
                    if 'timestamp' in data_point:
                        self.timestamps.append(data_point['timestamp'])
                    
                elif data_point['type'] == 'target':
                    self.target = data_point['target']
                    # Mettre à jour le point cible dans le graphique de trajectoire
                    self.target_point.set_data([self.target[0]], [self.target[1]])
                    
                    # Note: Nous ne modifions plus les limites ici car elles sont fixées
                    # par world_bounds
                    
                self.data_queue.task_done()
            except queue.Empty:
                break
        
        # Si aucune donnée n'est disponible, retourner
        if not self.timestamps:
            return
            
        # Convertir les timestamps relatifs en secondes
        t_start = self.timestamps[0]
        rel_time = [(t - t_start) for t in self.timestamps]
        
        # Mettre à jour le graphique de trajectoire
        if self.trajectory:
            x_data = [p[0] for p in self.trajectory]
            y_data = [p[1] for p in self.trajectory]
            self.trajectory_line.set_data(x_data, y_data)
            
            # Position actuelle (dernier point)
            current_pos = self.trajectory[-1]
            self.trajectory_point.set_data([current_pos[0]], [current_pos[1]])
            
            # Point de départ (une seule fois)
            if not self.start_set:
                start = self.trajectory[0]
                self.start_point.set_data([start[0]], [start[1]])
                self.start_set = True

        
        # Mettre à jour le graphique de coût
        if self.cost_history:
            self.cost_line.set_data(rel_time, self.cost_history)
            if len(self.cost_history) > 1:
                self.ax_cost.set_xlim(0, max(rel_time[-1], 1.0))  # Au moins 1 seconde affichée
                # Définir une limite minimale pour l'axe y pour éviter les problèmes avec des coûts très faibles
                ymax = max(max(self.cost_history) * 1.1, 0.1)
                self.ax_cost.set_ylim(0, ymax)
        
        # Mettre à jour le graphique des vitesses des roues
        if self.wheel_speeds:
            left_speeds = [ws[0] for ws in self.wheel_speeds]
            right_speeds = [ws[1] for ws in self.wheel_speeds]
            self.left_wheel_line.set_data(rel_time, left_speeds)
            self.right_wheel_line.set_data(rel_time, right_speeds)
            
            if len(self.wheel_speeds) > 1:
                self.ax_wheels.set_xlim(0, max(rel_time[-1], 1.0))  # Au moins 1 seconde affichée
                min_speed = min(min(left_speeds or [0]), min(right_speeds or [0]))
                max_speed = max(max(left_speeds or [0]), max(right_speeds or [0]))
                margin = max((max_speed - min_speed) * 0.1, 0.1)  # Marge d'au moins 0.1
                self.ax_wheels.set_ylim(min_speed - margin, max_speed + margin)
        
        # Mettre à jour le graphique du gradient
        if self.gradients:
            grad_left = [g[0] for g in self.gradients]
            grad_right = [g[1] for g in self.gradients]
            self.grad_line_left.set_data(rel_time, grad_left)
            self.grad_line_right.set_data(rel_time, grad_right)
            
            if len(self.gradients) > 1:
                self.ax_gradient.set_xlim(0, max(rel_time[-1], 1.0))  # Au moins 1 seconde affichée
                all_grads = grad_left + grad_right
                if all_grads:
                    min_grad = min(all_grads)
                    max_grad = max(all_grads)
                    margin = max((max_grad - min_grad) * 0.1, 0.1)  # Marge d'au moins 0.1
                    self.ax_gradient.set_ylim(min_grad - margin, max_grad + margin)
        
        # Mise à jour distances (erreurs)
        if self.distances:
            err_x = [d[0] for d in self.distances]
            err_y = [d[1] for d in self.distances]
            err_theta = [d[2] for d in self.distances]
            
            # Erreurs X et Y
            self.distance_x_line.set_data(rel_time, err_x)
            self.distance_y_line.set_data(rel_time, err_y)
            if len(self.distances) > 1:
                self.ax_distance_xy.set_xlim(0, max(rel_time[-1], 1.0))
                all_errs_xy = err_x + err_y
                if all_errs_xy:
                    min_err = min(all_errs_xy)
                    max_err = max(all_errs_xy)
                    margin = max((max_err - min_err) * 0.1, 0.1)
                    self.ax_distance_xy.set_ylim(min_err - margin, max_err + margin)
            
            # Erreur θ
            self.distance_theta_line.set_data(rel_time, err_theta)
            if len(self.distances) > 1:
                self.ax_distance_theta.set_xlim(0, max(rel_time[-1], 1.0))
                if err_theta:
                    min_err_theta = min(err_theta)
                    max_err_theta = max(err_theta)
                    margin_theta = max((max_err_theta - min_err_theta) * 0.1, 0.1)
                    self.ax_distance_theta.set_ylim(min_err_theta - margin_theta, max_err_theta + margin_theta)
        
        # Ajuster la disposition pour s'assurer que tout est visible
        self.fig.canvas.flush_events()
    
    def start(self):
        """Démarre l'affichage des graphiques dans le thread principal"""
        if not self.running:
            self.running = True
            # Initialiser les graphiques seulement lors du démarrage
            self.setup_plots()
    
    def stop(self):
        """Arrête l'affichage"""
        if self.running:
            self.running = False
            if self.fig:
                plt.close(self.fig)
    
    def add_data_point(self, position, wheel_speeds, gradient, cost, distances):
        """
        Ajoute un point de données au moniteur via la queue
        
        Args:
            position (list): Position [x, y, theta] du robot
            wheel_speeds (list): Vitesses [gauche, droite] des roues
            gradient (list): Gradient [gauche, droite] pour l'apprentissage
            cost (float): Valeur de la fonction de coût
            distances (list): Distances/erreurs [x, y, theta] par rapport à la cible
        """
        if self.running:
            data = {
                'type': 'data',
                'position': position,
                'wheel_speeds': wheel_speeds,
                'gradient': gradient,
                'cost': cost,
                'distances': distances,
                'timestamp': time.time()
            }
            self.data_queue.put(data)
    
    def set_target(self, target):
        """
        Définit la position cible via la queue
        
        Args:
            target (list): Position cible [x, y, theta]
        """
        if self.running:
            self.data_queue.put({
                'type': 'target',
                'target': target
            })
            
    def save_plots(self, base_filename="robot_monitoring", timestamp=True):
        """
        Sauvegarde les graphiques actuels sous forme de fichiers PNG
        
        Args:
            base_filename (str): Nom de base pour les fichiers
            timestamp (bool): Si True, ajoute un timestamp au nom de fichier
        """
        if timestamp:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            base_filename = f"{base_filename}_{timestamp_str}"
        
        # Créer le dossier de résultats s'il n'existe pas
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Sauvegarder la figure complète
        if self.fig:
            plt.figure(self.fig.number)
            save_path = os.path.join(results_dir, f"{base_filename}_all.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

# Classe adaptateur pour intégrer le monitoring aux différentes implémentations de l'apprentissage
class RobotMonitorAdapter:
    """
    Adaptateur pour connecter le module de monitoring aux différentes
    implémentations d'apprentissage (Python natif ou PyTorch)
    """
    
    def __init__(self, alpha_values=None, world_bounds=(-10, 10, -10, 10)):
        """
        Initialise l'adaptateur
        
        Args:
            alpha_values (list): Les valeurs alpha utilisées pour la normalisation [x, y, theta]
            world_bounds (tuple): Limites du monde (x_min, x_max, y_min, y_max)
        """
        # Créer le moniteur avec les limites du monde personnalisées
        self.monitor = RobotMonitor(world_bounds=world_bounds)
        
        self.alpha = alpha_values or [1/6, 1/6, 1/(math.pi)]  # Valeurs par défaut
        self.last_position = None
        self.last_wheel_speeds = None
        self.last_gradient = None
        self.last_target = None
        self.current_cost = 0.0
        
    def start_monitoring(self):
        """Démarre le monitoring"""
        self.monitor.start()
        
    def stop_monitoring(self):
        """Arrête le monitoring"""
        self.monitor.stop()
        
    def set_target(self, target):
        """
        Définit la position cible et met à jour le moniteur
        
        Args:
            target (list): Position cible [x, y, theta]
        """
        self.last_target = target
        self.monitor.set_target(target)
    
    def update(self, position, wheel_speeds, gradient=None, cost=None, add_noise=False):
        """
        Met à jour le moniteur avec les données actuelles
        
        Args:
            position (list): Position actuelle [x, y, theta]
            wheel_speeds (list): Vitesses des roues [gauche, droite]
            gradient (list, optional): Gradient [gauche, droite]
            cost (float, optional): Valeur de la fonction de coût
            add_noise (bool): Si True, ajoute un petit bruit aléatoire pour l'affichage
        """
        # Sauvegarder les dernières valeurs
        self.last_position = position
        self.last_wheel_speeds = wheel_speeds
        
        if gradient is not None:
            self.last_gradient = gradient
        
        # Calculer le coût si non fourni
        if cost is None and self.last_target is not None:
            self.current_cost = self._calculate_cost(position, self.last_target)
        else:
            self.current_cost = cost or 0.0
        
        # Calculer les distances/erreurs par rapport à la cible
        distances = self._calculate_distances(position, self.last_target or [0, 0, 0])
        
        # Ajouter du bruit pour l'affichage si demandé
        if add_noise and gradient is not None:
            gradient = [g + np.random.normal(0, 0.001) for g in gradient]
        
        # Mettre à jour le moniteur
        self.monitor.add_data_point(
            position=position,
            wheel_speeds=wheel_speeds,
            gradient=gradient or [0, 0],
            cost=self.current_cost,
            distances=distances
        )
    
    def _calculate_cost(self, position, target):
        """
        Calcule la fonction de coût
        
        Args:
            position (list): Position actuelle [x, y, theta]
            target (list): Position cible [x, y, theta]
            
        Returns:
            float: Valeur de la fonction de coût
        """
        # Fonction theta_s de l'implémentation originale
        def theta_s(x, y):
            return math.tanh(10.*x)*math.atan(1.*y)
        
        alpha_x = self.alpha[0]
        alpha_y = self.alpha[1]
        alpha_theta = self.alpha[2]
        
        return (alpha_x * alpha_x * (position[0] - target[0])**2 + 
                alpha_y * alpha_y * (position[1] - target[1])**2 + 
                alpha_theta * alpha_theta * (position[2] - target[2] - 
                                         theta_s(position[0], position[1]))**2)
    
    def _calculate_distances(self, position, target):
        """
        Calcule les distances/erreurs par rapport à la cible
        
        Args:
            position (list): Position actuelle [x, y, theta]
            target (list): Position cible [x, y, theta]
            
        Returns:
            list: Distances/erreurs [x, y, theta]
        """
        # Pour l'erreur theta, on considère l'angle minimal (cyclic)
        theta_diff = (position[2] - target[2]) % (2 * math.pi)
        if theta_diff > math.pi:
            theta_diff -= 2 * math.pi
            
        return [
            position[0] - target[0],  # Erreur en X
            position[1] - target[1],  # Erreur en Y
            theta_diff                # Erreur en theta
        ]
    
    def save_results(self, base_filename="robot_results"):
        """
        Sauvegarde les résultats du monitoring
        
        Args:
            base_filename (str): Nom de base pour les fichiers
        """
        # Sauvegarder les graphiques
        self.monitor.save_plots(base_filename)