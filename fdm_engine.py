import numpy as np

def interpolate_fluid_temp(times, temps, residence_multiplier, dt=0.1):
    total_time = times[-1]
    if isinstance(residence_multiplier, str):
        factor = 0.85 if "低" in residence_multiplier or "A" in residence_multiplier or "Low" in residence_multiplier else 0.50
    else:
        factor = float(residence_multiplier)
    adjusted_times = np.array(times) * factor
    
    interp_times = np.arange(0, adjusted_times[-1] + dt, dt)
    interp_temps = np.interp(interp_times, adjusted_times, temps)
    
    return interp_times, interp_temps, dt

def solve_sphere_fdm(r_mm, alpha, k, h, fluid_temps, dt=0.1):
    R = r_mm / 1000.0
    N = 10
    dr = R / N
    Bi = h * dr / k
    # 表面节点严苛的稳定性判据：1 - 2*Fo*(1 + Bi*(1+1/N)) >= 0
    max_Fo = 1.0 / (2.0 * (1.0 + Bi * (1.0 + 1.0/N))) * 0.90
    
    Fo = alpha * dt / (dr**2)
    
    sub_steps = 1
    if Fo >= max_Fo:
        sub_steps = int(np.ceil(Fo / max_Fo))
        Fo = Fo / sub_steps
    
    T = np.ones(N + 1) * fluid_temps[0]
    center_temps = [T[0]]
    grid_history = [np.copy(T)]
    
    for i in range(1, len(fluid_temps)):
        Tf_prev = fluid_temps[i-1]
        Tf_curr = fluid_temps[i]
        
        for s in range(1, sub_steps + 1):
            Tif = Tf_prev + (Tf_curr - Tf_prev) * (s / sub_steps)
            T_new = np.copy(T)
            
            # Center node singularity handled with L'Hopital rule
            T_new[0] = T[0] + 6 * Fo * (T[1] - T[0])
            
            # Interior nodes
            for j in range(1, N):
                T_new[j] = T[j] + Fo * ((1 - 1.0/j)*T[j-1] - 2*T[j] + (1 + 1.0/j)*T[j+1])
                
            # Surface node
            T_new[N] = T[N] + 2 * Fo * (T[N-1] - T[N] + Bi * (Tif - T[N]) * (1 + 1.0/N))
            
            T = T_new
        center_temps.append(T[0])
        grid_history.append(np.copy(T))
        
    return np.array(center_temps), np.array(grid_history)

def solve_cube_fdm(l_mm, alpha, k, h, fluid_temps, dt=0.1):
    L = l_mm / 1000.0
    N = 10
    dx = L / N
    Bi = h * dx / k
    # 角落节点受到3个面同时对流换热，稳定性要求极高：1 - 6*Fo*(1 + Bi) >= 0
    max_Fo = 1.0 / (6.0 * (1.0 + Bi)) * 0.90
    
    Fo = alpha * dt / (dx**2)
    
    sub_steps = 1
    if Fo >= max_Fo:
        sub_steps = int(np.ceil(Fo / max_Fo))
        Fo = Fo / sub_steps
    
    T = np.ones((N+1, N+1, N+1)) * fluid_temps[0]
    center_temps = [T[0,0,0]]
    grid_history = [np.copy(T[:,:,0])] # 中截面 z=0
    
    for i in range(1, len(fluid_temps)):
        Tf_prev = fluid_temps[i-1]
        Tf_curr = fluid_temps[i]
        
        for s in range(1, sub_steps + 1):
            Tif = Tf_prev + (Tf_curr - Tf_prev) * (s / sub_steps)
            T_new = np.copy(T)
            
            # Interior
            T_new[1:N, 1:N, 1:N] = T[1:N, 1:N, 1:N] + Fo * (
                T[0:N-1, 1:N, 1:N] + T[2:N+1, 1:N, 1:N] +
                T[1:N, 0:N-1, 1:N] + T[1:N, 2:N+1, 1:N] +
                T[1:N, 1:N, 0:N-1] + T[1:N, 1:N, 2:N+1] -
                6 * T[1:N, 1:N, 1:N]
            )
            
            # Boundaries
            for x in range(N+1):
                for y in range(N+1):
                    for z in range(N+1):
                        if x == 0 or x == N or y == 0 or y == N or z == 0 or z == N:
                            tx_m = T[x-1,y,z] if x>0 else T[1,y,z]
                            tx_p = T[x+1,y,z] if x<N else T[N-1,y,z] + 2*Bi*(Tif - T[N,y,z])
                            
                            ty_m = T[x,y-1,z] if y>0 else T[x,1,z]
                            ty_p = T[x,y+1,z] if y<N else T[x,N-1,z] + 2*Bi*(Tif - T[x,N,z])
                            
                            tz_m = T[x,y,z-1] if z>0 else T[x,y,1]
                            tz_p = T[x,y,z+1] if z<N else T[x,y,N-1] + 2*Bi*(Tif - T[x,y,N])
                            
                            T_new[x,y,z] = T[x,y,z] + Fo * (tx_m + tx_p + ty_m + ty_p + tz_m + tz_p - 6*T[x,y,z])
            T = T_new
        center_temps.append(T[0,0,0])
        grid_history.append(np.copy(T[:,:,0]))
        
    return np.array(center_temps), np.array(grid_history)

def calculate_lethality(temps, dt, ph, t_ref, z_val):
    PU = 0.0
    F0 = 0.0
    for T in temps:
        if ph < 4.6:
            if T > 60.0:
                PU += 10**((T - t_ref)/z_val) * (dt / 60.0)
        else:
            if T > 100.0:
                F0 += 10**((T - t_ref)/z_val) * (dt / 60.0)
    return PU, F0
