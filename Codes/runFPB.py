from potentiel import *
from doubleSlit_FPB_CN import *
from createAnimations import *
from time import time
import tracemalloc

if __name__ == "__main__":
    L = 8 # Grandeur de la simulation (de la boîte).
    Dy = 0.05 # Pas d'espace.
    Dt = Dy**2/4 # Pas de temps.
    Nx = int(L/Dy) + 1 # Grandeur du grillage en x.
    Ny = int(L/Dy) + 1 # Grandeur du grillage en y.
    Nt = 500 # Nombre de points de temps.
    v = np.zeros((Ny,Ny), complex)  # Première définition du potentiel.
    k = 5*np.pi # Nombre d'ondes dans la boîte.

    # Position initial du faisceau d'électrons.
    x0 = L/5
    y0 = L/2
        
    Ni = (Nx-2)*(Ny-2)  # Nombre d'inconnus v[i,j], i = 1,...,Nx-2, j = 1,...,Ny-2

    j0, j1, i0, i1, i2, i3, v, w = potentielSlits(Dy, Ny, L, k, y0)

    # v_abs = potentiel_absorbant(x, y, L, v, d_abs=2, strength=100) # Fucking instable
    # v += v_abs
    
    mat_t = time()
    A, M = buildMatrix(Ni, Nx, Ny, Dy, Dt, v)
    mat_t = time() - mat_t
    print(f'Temps d\'exécution de création de la matrice: : {mat_t*1000:.2f} ms')
    
    tracemalloc.start()
    solvemat_t = time()
    mod_psis, initial_norm = solveMatrix(A, M, L, Nx, Ny, Ni, Nt, x0, y0, Dy)
    solvemat_t = time() - solvemat_t
    print(f'Temps d\'exécution de résolution de la matrice: : {solvemat_t*1000:.2f} ms')
    current, peak = tracemalloc.get_traced_memory()
    print(f"Utilisation actuelle : {current / 10**6} Mo; Pic : {peak / 10**6} Mo")
    tracemalloc.stop()
    

    final_psi = diffractionPatron(mod_psis, L, Ny)
    final_norm = np.sum(np.abs(final_psi)**2) * Dy * Dy

    # print(f"Probabilité totale initiale : {initial_norm}")
    # print(f"Probabilité totale finale : {final_norm}")
    print(f"Fonction d'onde bien normalisée : {0.95 <= initial_norm <= 1.05 and 0.95 <= final_norm <= 1.05}")

    animation = makeAnimationForSlits(mod_psis, v, L, Nt)

