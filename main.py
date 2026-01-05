import numpy as np
from core.optmizer import FCCPMSO

def sphere_benchmark(x):
    """Função Sphere: f(x) = sum(x^2). Mínimo global em 0."""
    fitness = np.sum(x**2)
    violations = 0 # Sem restrições para este teste básico
    return fitness, violations

if __name__ == "__main__":
    # Configurações do Artigo: 4 enxames, 100 partículas, 140 iterações [cite: 2483]
    optimizer = FCCPMSO(
        n_swarms=4, 
        n_particles=100, 
        dim=10, 
        iterations=140, 
        bounds=(-5.12, 5.12)
    )
    
    print("Iniciando FC-CPMSO...")
    best_params, best_score = optimizer.run(sphere_benchmark)
    
    print("-" * 30)
    print(f"Otimização concluída!")
    print(f"Melhor Fitness encontrado: {best_score}")
    print(f"Parâmetros: {best_params}")