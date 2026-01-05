import numpy as np

def apply_velocity_clamping(velocity, x_min, x_max, beta=0.5):
    """
    Aplica o Clamping de velocidade modificado (Eq. 3 a 6).
    Inclui o tratamento para partículas em estado estacionário.
    """
    # Calcula os limites de velocidade baseados no range do espaço (Eq. 4)
    range_space = np.abs(x_max - x_min)
    v_min = -(range_space * beta)
    
    # Vmax é ajustado estocasticamente conforme o artigo (Eq. 5)
    v_max = x_max * np.random.uniform(0.8, 1.0)
    
    # Tratamento para estado estacionário (V = 0) (Eq. 3 e 6)
    # Se a velocidade for zero, atribui um pequeno impulso aleatório
    is_stationary = (velocity == 0)
    p_v = range_space * np.random.uniform(0.05, 0.1)
    
    # Aplica as regras da Equação 6
    new_v = np.copy(velocity)
    new_v = np.where(velocity < v_min, v_min, new_v)
    new_v = np.where(velocity > v_max, v_max, new_v)
    new_v = np.where(is_stationary, p_v, new_v)
    
    return new_v

def apply_position_boundary(position, velocity, x_min, x_max):
    """
    Aplica as condições de contorno de Damping (Eq. 7 e 8).
    Reposiciona a partícula e ajusta sua velocidade ao atingir as bordas.
    """
    new_pos = np.copy(position)
    new_vel = np.copy(velocity)
    
    # Range para o cálculo do reset de velocidade
    range_space = np.abs(x_max - x_min)
    v_min = -(range_space * 0.5) # Beta padrão do artigo
    v_max = x_max * 0.9 # Média do U(0.8, 1.0)
    
    # Regra de reposicionamento (Eq. 7)
    # Se X < Xmin -> X = Xmin; Se X > Xmax -> X = Xmax
    out_lower = position < x_min
    out_upper = position > x_max
    
    new_pos[out_lower] = x_min[out_lower]
    new_pos[out_upper] = x_max[out_upper]
    
    # Regra de correção de velocidade (Damping - Eq. 8)
    # Se atingir a borda, a velocidade é refletida/ajustada estocasticamente
    new_vel[out_lower] = -v_min[out_lower] * np.random.uniform(0, 1)
    new_vel[out_upper] = -v_max[out_upper] * np.random.uniform(0, 1)
    
    return new_pos, new_vel