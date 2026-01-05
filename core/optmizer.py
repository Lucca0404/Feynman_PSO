import numpy as np
from .swarm import Swarm
from utils.boundary_conditions import apply_velocity_clamping, apply_position_boundary

class FCCPMSO:
    def __init__(self, n_swarms=4, n_particles=100, dim=2, iterations=140, bounds=(-10, 10)):
        self.K, self.P, self.D, self.T = n_swarms, n_particles, dim, iterations
        self.bounds = (np.full(dim, bounds[0]), np.full(dim, bounds[1]))
        # Inicializa os enxames (Estrutura Multi-Swarm) [cite: 1812]
        self.swarms = [Swarm(k, n_particles, dim, bounds) for k in range(n_swarms)]
        self.s_best_pos = None
        self.s_best_fit = float('inf')

    def run(self, objective_func):
        # Avaliação inicial [cite: 1797]
        for swarm in self.swarms:
            swarm.evaluate_swarm(objective_func)
            if swarm.g_best_fit < self.s_best_fit:
                self.s_best_fit = swarm.g_best_fit
                self.s_best_pos = np.copy(swarm.g_best_pos)

        # Loop de iterações (T) [cite: 1799]
        for t in range(self.T):
            for k in range(self.K):
                for i in range(self.P):
                    p = self.swarms[k].get_particle(i)
                    
                    # 1. Atualiza Velocidade com PAF e TMR [cite: 2027, 2038]
                    # (Aqui você chama a lógica de velocidade que discutimos antes)
                    # self.update_velocity(k, i, t) 
                    
                    # 2. Aplica Clamping de Velocidade (Eq. 6) [cite: 1778]
                    p.velocity = apply_velocity_clamping(p.velocity, self.bounds[0], self.bounds[1])
                    
                    # 3. Atualiza Posição e aplica Damping (Eq. 7-8) [cite: 1782, 1788]
                    p.position += p.velocity
                    p.position, p.velocity = apply_position_boundary(
                        p.position, p.velocity, self.bounds[0], self.bounds[1]
                    )

            # Reavalia e atualiza memórias globais [cite: 1805]
            for swarm in self.swarms:
                swarm.evaluate_swarm(objective_func)
                if swarm.g_best_fit < self.s_best_fit:
                    self.s_best_fit = swarm.g_best_fit
                    self.s_best_pos = np.copy(swarm.g_best_pos)
            
            if t % 10 == 0:
                print(f"Iteração {t}: Melhor Fitness = {self.s_best_fit}")

        return self.s_best_pos, self.s_best_fit