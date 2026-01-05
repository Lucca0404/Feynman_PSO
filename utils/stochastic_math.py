import numpy as np

def get_stochastic_inertia(t, T):
    """Calcula a inércia estocástica baseada nas Equações 20-24 [cite: 537-550]"""
    w_min, w_max = 0.0, 1.0
    r = np.random.rand()
    expanded = r > 0.7 # Threshold de 0.7 [cite: 556]
    
    # Divisão em 3 estágios de iterações [cite: 539]
    if t < T * 0.33:
        return np.random.uniform(0.5, 1.0) if not expanded else np.random.uniform(0.2, 1.0)
    elif t < T * 0.66:
        return np.random.uniform(0.2, 0.8) if not expanded else np.random.uniform(0.1, 0.9)
    else:
        return np.random.uniform(0.0, 0.5) if not expanded else np.random.uniform(0.0, 0.8)

def get_poli_coefficients(nc, active_mask):
    """Ajusta C1, Cc, Cg, Cs para garantir convergência (Eq. 34-37) [cite: 616-644]"""
    # A soma dos coeficientes deve estar na região parabólica [cite: 658, 661]
    c_list = [0.0, 0.0, 0.0, 0.0]
    if nc == 0: return c_list
    
    # Distribuição sugerida para manter a soma <= 2.0 [cite: 661]
    base_c = 2.0 / nc 
    for i in range(4):
        if active_mask[i]:
            c_list[i] = np.random.uniform(0, base_c)
    return c_list