import numpy as np
import psutil
from potentiel import *
from doubleSlit_FPB_CN import *
from createAnimations import *

"""
Regarder la variation de la mémoire en fonction de différents paramètres.
"""
    
dy_values = np.linspace(0.02, 0.07, 10)
    
for dy in dy_values:
        
        Dt = dy**2/4 # Pas de temps.
        L = 8 # Grandeur de la simulation (de la boîte).
        Nx = int(L/dy) + 1 # Grandeur du grillage en x.
        Ny = int(L/dy) + 1 # Grandeur du grillage en y.
        Nt = 500 # Nombre de points de temps.
        v = np.zeros((Ny,Ny), complex)  # Première définition du potentiel.
        k = 5*np.pi # Nombre d'ondes dans la boîte.
        
        # Position initial du faisceau d'électrons.
        x0 = L/5
        y0 = L/2
        
        Ni = (Nx-2)*(Ny-2)  # Nombre d'inconnus v[i,j], i = 1,...,Nx-2, j = 1,...,Ny-2
        
        j0, j1, i0, i1, i2, i3, v, w = potentielSlits(dy, Ny, L, k, y0)
        
        A, M = buildMatrix(Ni, Nx, Ny, dy, Dt, v)
        
        mod_psis, initial_norm = solveMatrix(A, M, L, Nx, Ny, Ni, Nt, x0, y0, dy)
        
        final_psi = diffractionPatron(mod_psis, L, Ny)
        final_norm = np.sum(np.abs(final_psi)**2) * dy * dy
        
        print(f"Fonction d'onde bien normalisée : {0.95 <= initial_norm <= 1.05 and 0.95 <= final_norm <= 1.05}")
        mem = psutil.Process().memory_info().rss / (1024 * 1024)
        print(dy)
        print(mem)