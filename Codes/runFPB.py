from potentiel import potentielSlits, potentiel_cristal
from doubleSlit_FPB_CN import *
from createAnimations import *
from time import time
import tracemalloc
import os
import numpy as np
import matplotlib.pyplot as plt

convergence_calculated = False

if __name__ == "__main__":

    fact_ar = np.array([0.05], dtype=np.double); # np.array([0.0100, 0.02, 0.03, 0.04, 0.05], dtype=np.double); 
    mem_ar=np.zeros(fact_ar.size,dtype=np.double)
    d_ar=np.zeros(fact_ar.size,dtype=np.double)

    ci = -1
    for fact in fact_ar:
        ci += 1
        print(f"Pas de discrétisation : dx=dy={fact}")
        d_ar[ci] = fact

        L = 30 # Grandeur de la simulation (de la boîte).
        sigma = L/20 # Amplitude du paquet d'onde
        Dy = fact # Pas d'espace.
        Dt = (Dy**2) # Pas de temps.
        extract_frac = 0.85

        x_extract = extract_frac * L

        
        n_cells = 10  # Une maille = 1 atome, on place un puits gaussien par maille. On génère (2n_cells+1) puits gaussiens
        h_bar = 1 
        m = 1    
        a = L/(2*n_cells+1)  # np.pi * 2 /k        # hauteur totale de chaque fente

        k = (np.pi / a) # k = 2pi/lambda 

        E_k = (k**2*h_bar**2)/(2*m)    # Énergie cinétique

        v_g = h_bar * k / m

        # Paramètres des fentes    
        v0 = 0.5 * E_k 

        y0 = L/2
        x0 = 3*sigma

        T = (x_extract - x0) / (v_g*1.1) # Temps total de simulation.

        Nx = int(L/Dy) + 1 # Grandeur du grillage en x.
        Ny = int(L/Dy) + 1 # Grandeur du grillage en y.
        print(f"Taille matrice : {Nx*Ny} éléments")
        Nt = int(T / Dt) # Nombre de points de temps.
        print(f"Nombre de points de temps : {Nt}")
        v = np.zeros((Ny,Ny), complex) 

        ### Nt * Dt = t = L/v_g = L * m / h_bar * k sous le modèle des électrons quasi-libres v_g = h_bar * k /m ###

        #s = a * 3   # distance entre centres de fentes
        #w = a       # épaisseur mur

        
        sigma_v = a/10   # largeur des puits gaussiens 
    
        # Position initial du faisceau d'électrons.

        Ni = (Nx-2)*(Ny-2)  # Nombre d'inconnus v[i,j], i = 1,...,Nx-2, j = 1,...,Ny-2

        #j0, j1, i0, i1, i2, i3, v, x_fentes = potentielSlits(Dy, Ny, L, y0, s, w, v0, a)

        v = potentiel_cristal(Dy, Nx, Ny, a, sigma_v, v0, n_cells)

        # position du plan d'atomes en x
        x_fentes = L/2

        # on prend toute la tranche y=1..Ny-2 pour la cumulation
        j0, j1 = 1, Ny-2
        # on fixe les indices x au bord intérieur (pas critique pour l'anim)
        i0, i1, i2, i3 = 1, 1, Nx-2, Ny-2

        mat_t = time()
        A, M = buildMatrix(Ni, Nx, Ny, Dy, Dt, v)
        mem = 8 * M.nnz + 8 * A.nnz
        print(f"Mémoire utilisée pour les matrices : {mem/10**6:.2f} Mo")
        mat_t = time() - mat_t
        print(f'Temps d\'exécution de création de la matrice: : {mat_t*1000:.2f} ms')
        
        tracemalloc.start()
        solvemat_t = time()
        mod_psis, initial_norm, norms = solveMatrix(A, M, L, Nx, Ny, Ni, Nt, x0, y0, Dy, k, sigma) # mod_psis est une liste de matrices (𝑁x−2)x(Ny-2)
        solvemat_t = time() - solvemat_t
        print(f'Temps d\'exécution de résolution de la matrice: : {solvemat_t*1000:.2f} ms')
        current, peak = tracemalloc.get_traced_memory()
        print(f"Utilisation actuelle : {current / 10**6} Mo; Pic : {peak / 10**6} Mo")
        tracemalloc.stop()

        # On calcul la mémoire ici
        M_csr = M.tocsr()
        mem_ar[ci] = 8 * M_csr.nnz

    # if not convergence_calculated:       # Décommenter pour calculer l'erreur de convergence
    #     dy_list = [0.04, 0.08, 0.16]  
    #     errors_l2, orders_l2 = convergence_erreur(L, T, x0, y0, k, dy_list, a, s, sigma, w, v0)
    #     convergence_calculated = True

    plt.figure(figsize=(8, 6))  
    plt.loglog(d_ar[::-1], mem_ar[::-1]/1024.0**3, '-o')
    plt.title('Exigences de mémoire')
    plt.xlabel('Pas $d_x=d_y$ [m]')
    plt.ylabel('Mémoire [Gb]')
    plt.grid(True, which="both", ls="--")  
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "mémoire.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()

     # on ne conserve que le puits central pour le calcul du temps d'arrivée
    distance_to_fentes = abs(x_fentes - x0)

    cumul_cible = distance_to_fentes * 1.5

    t_arrival = abs(cumul_cible) / v_g  # Temps pour atteindre x_center
    n0 = int(t_arrival / Dt)  # Convertir en pas de temps
    n0 = max(0, min(n0, Nt-1))  # S'assurer que n0 est dans les limites [0, Nt-1])
    
    print(f"Début cumul | n0 : {n0}") 


    # D : distance entre le plan des puits et l’écran
    D = abs(x_extract - x_fentes)

    final_psi = diffractionPatron(mod_psis, L, Ny, a, k, D, n0, extract_frac, x0, Dt, v_g)
    final_norm = np.sum(np.abs(mod_psis[-1])**2) * Dy * Dy

    print(f"Norme finale : {final_norm}")

    #assert 0.95 <= initial_norm <= 1.05 and 0.95 <= final_norm <= 1.05

    print(f"Probabilité totale initiale : {initial_norm}")
    print(f"Probabilité totale finale : {final_norm}")

    plt.figure(figsize=(10, 6))
    time_steps = np.arange(len(norms)) * Dt 
    plt.plot(time_steps, norms, label='Norme de ψ')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Norme théorique (1)')
    plt.xlabel('Temps (s)')
    plt.ylabel('Norme (|ψ|² intégré)')
    plt.title('Évolution de la norme de la fonction d\'onde')
    plt.ylim(0.8, 1.0)
    plt.legend()
    plt.grid(True)
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "norm_evolution.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    time_steps = np.arange(len(norms)) * Dt
    plt.semilogy(time_steps, norms, label='Norme de ψ (échelle log)')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Norme théorique (1)')
    plt.xlabel('Temps (s)')
    plt.ylabel('Norme (|ψ|² intégré)')
    plt.title("Évolution de la norme de la fonction d'onde (log)")
    plt.ylim(bottom=min(norms)*0.8, top=max(norms)*1.1)
    plt.legend()
    plt.grid(True, which='both', ls='--')
    # Sauvegarde
    output_file_log = os.path.join(output_dir, "norm_evolution_log.png")
    plt.savefig(output_file_log, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Fonction d'onde bien normalisée : {0.95 <= initial_norm <= 1.05 and 0.95 <= final_norm <= 1.05}")

    animation = makeAnimationForSlits(
        mod_psis, v, L, Nt, n0, v_g, Dt,
        x0, Dy, extract_frac, x_fentes, x_extract, D, sigma
    )
