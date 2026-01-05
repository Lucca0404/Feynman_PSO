import numpy as np

class Particle:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.velocity = np.zeros(dim)
        self.prev_position = np.copy(self.position)
        self.direction = np.zeros(dim) # Trajetória (gradiente) [cite: 751]
        
        self.fitness = float('inf')
        self.violations = 0
        
        # Memória individual (Local Best) [cite: 32, 240]
        self.p_best_pos = np.copy(self.position)
        self.p_best_fit = float('inf')
        self.p_best_violations = float('inf')

    def update_pbest(self):
        # Regras de viabilidade de Deb [cite: 966, 108]
        if (self.violations < self.p_best_violations) or \
           (self.violations == self.p_best_violations and self.fitness < self.p_best_fit):
            self.p_best_pos = np.copy(self.position)
            self.p_best_fit = self.fitness
            self.p_best_violations = self.violations