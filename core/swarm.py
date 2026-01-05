import numpy as np
from .particle import Particle

class Swarm:
    def __init__(self, swarm_id, n_particles, dim, bounds):
        """
        Representa um enxame individual dentro da estrutura Multi-Swarm.
        Organizado conforme a estrutura de blocos do CUDA (Fig. 12).
        """
        self.swarm_id = swarm_id
        # Inicializa as partículas conforme a Figura 2 do artigo
        self.particles = [Particle(dim, bounds) for _ in range(n_particles)]
        
        # Memória Global do Enxame (gbest) - Categoria B da Fig. 3
        self.g_best_pos = None
        self.g_best_fit = float('inf')
        self.g_best_violations = float('inf')

    def update_gbest(self):
        """
        Analisa o enxame e atualiza o Global Best baseado na Regra de Deb.
        Prioriza: 1. Menor número de violações; 2. Menor Fitness.
        """
        for p in self.particles:
            # Lógica de comparação baseada na viabilidade (Deb's Rule)
            if (p.p_best_violations < self.g_best_violations) or \
               (p.p_best_violations == self.g_best_violations and p.p_best_fit < self.g_best_fit):
                self.g_best_pos = np.copy(p.p_best_pos)
                self.g_best_fit = p.p_best_fit
                self.g_best_violations = p.p_best_violations

    def get_particle(self, index):
        """
        Retorna a partícula pelo índice. 
        Essencial para a topologia de 'Counterpart Particles' (Eq. 9-11).
        """
        return self.particles[index]

    def evaluate_swarm(self, objective_func):
        """
        Ciclo de avaliação de cada partícula (Algoritmo 2).
        """
        for p in self.particles:
            # A função objetivo deve retornar (fitness, violations)
            p.fitness, p.violations = objective_func(p.position)
            p.update_pbest()
        
        # Após avaliar todos, atualiza a memória coletiva do enxame
        self.update_gbest()