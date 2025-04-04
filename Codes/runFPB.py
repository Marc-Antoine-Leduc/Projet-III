from potentiel import *
from doubleSlit_FPB_CN import *
from createAnimations import *
from time import time
import tracemalloc

if __name__ == "__main__":

    fact_ar = np.array([0.05], dtype=np.double); # Matrice pleine
    mem_ar=np.zeros(fact_ar.size,dtype=np.double)
    d_ar=np.zeros(fact_ar.size,dtype=np.double)

    ci = -1
    for fact in fact_ar:
        ci += 1
        print(f"Pas de discrétisation : dx=dy={fact}")
        d_ar[ci] = fact

        L = 20 # Grandeur de la simulation (de la boîte).
        Dy = fact # Pas d'espace.
        Dt = Dy**2/4 # Pas de temps.
        Nx = int(L/Dy) + 1 # Grandeur du grillage en x.
        Ny = int(L/Dy) + 1 # Grandeur du grillage en y.
        Nt = 1000 # Nombre de points de temps.
        v = np.zeros((Ny,Ny), complex)  # Première définition du potentiel.

        ### Nt * Dt = t = L/v_g = L * m / h_bar * k sous le modèle des électrons quasi-libres v_g = h_bar * k /m ###

        h_bar = 1 
        m = 1    

        k = 15*np.pi # (L * m) / (h_bar * Nt * Dt)

        v_g = h_bar * k / m

        # Position initial du faisceau d'électrons.
        x0 = 3 
        y0 = L/2

        t_arrival = (L - x0) / v_g

        n0 = int(t_arrival / Dt)

            
        Ni = (Nx-2)*(Ny-2)  # Nombre d'inconnus v[i,j], i = 1,...,Nx-2, j = 1,...,Ny-2

        v, s, a = potentielSlits(Dy, Ny, L, y0)

        # v_abs = potentiel_absorbant(x, y, L, v, d_abs=2, strength=100) # Fucking instable
        # v += v_abs
        
        mat_t = time()
        A, M = buildMatrix(Ni, Nx, Ny, Dy, Dt, v)
        mem = 8 * M.nnz + 8 * A.nnz
        print(f"Mémoire utilisée pour les matrices : {mem/10**6:.2f} Mo")
        mat_t = time() - mat_t
        print(f'Temps d\'exécution de création de la matrice: : {mat_t*1000:.2f} ms')
        
        tracemalloc.start()
        solvemat_t = time()
        mod_psis, initial_norm, norms = solveMatrix(A, M, L, Nx, Ny, Ni, Nt, x0, y0, Dy)
        solvemat_t = time() - solvemat_t
        print(f'Temps d\'exécution de résolution de la matrice: : {solvemat_t*1000:.2f} ms')
        current, peak = tracemalloc.get_traced_memory()
        print(f"Utilisation actuelle : {current / 10**6} Mo; Pic : {peak / 10**6} Mo")
        tracemalloc.stop()

        # On calcul la mémoire ici
        M_csr = M.tocsr()
        mem_ar[ci] = 8 * M_csr.nnz

    plt.loglog(d_ar[::-1],mem_ar[::-1]/1024.0**3,'-o')
    plt.title('Exigences de mémoire')
    plt.xlabel('Pas $d_x=d_y$ [m]')
    plt.ylabel('Mémoire [Gb]')
    plt.show()   
    

    n0 = len(mod_psis) // 2
    extract_frac = 0.85
    x_extract = extract_frac * L
    D = abs(x_extract - 6)

    final_psi = diffractionPatron(mod_psis, L, Ny, s, a, k, D, n0, extract_frac)
    final_norm = np.sum(np.abs(mod_psis[-1])**2) * Dy * Dy

    print(f"Norme finale : {final_norm}")

    #Verification de la norme.
    # assert 0.95 <= initial_norm <= 1.05 and 0.95 <= final_norm <= 1.05

    # print(f"Probabilité totale initiale : {initial_norm}")
    # print(f"Probabilité totale finale : {final_norm}")

    plt.figure(figsize=(10, 6))
    time_steps = np.arange(len(norms)) * Dt 
    plt.plot(time_steps, norms, label='Norme de ψ')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Norme théorique (1)')
    plt.xlabel('Temps (s)')
    plt.ylabel('Norme (|ψ|² intégré)')
    plt.title('Évolution de la norme de la fonction d\'onde')
    plt.ylim(0.5, plt.ylim()[1]) 
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Fonction d'onde bien normalisée : {0.95 <= initial_norm <= 1.05 and 0.95 <= final_norm <= 1.05}")

    animation = makeAnimationForSlits(mod_psis, v, L, Nt, extract_frac)

