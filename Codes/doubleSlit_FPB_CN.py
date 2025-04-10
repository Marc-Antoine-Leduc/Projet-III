import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized
from scipy.sparse import lil_matrix, diags
from potentiel import potentielSlits
import os

def psi0(x, y, x0, y0, sigma, k):
    N = 1 / (sigma * np.sqrt(np.pi))  # Facteur de normalisation
    conditionInitiale = N * np.exp(-1/2*((x-x0)**2 + (y-y0)**2)/sigma**2) * np.exp(1j*k*(x-x0))
    return conditionInitiale

def buildMatrix(Ni, Nx, Ny, Dy, Dt, v):
    """
    Construit la matrice de système d'équations pour la méthode de 
    Cranck-Nicholson.

    Args :
        Ni (int) : Nombre de coefficients inconnus.
        Nx (int) : Grandeur du grillage en x.
        Ny (int) : Grandeur du grillage en y.
        Dy (float) : Pas en y.
        Dt (float) : Pas de temps.
        v (float) : Potentiel.

    Returns :
        A (ndarray) : Matrice à résoudre.
        M (ndarray) : Matrice à résoudre.
    """
    A = lil_matrix((Ni, Ni), dtype=complex)
    M = lil_matrix((Ni, Ni), dtype=complex)

    rx = -Dt/(2j*Dy**2)
    ry = -Dt/(2j*Dy**2)

    for k in range(Ni):     
        i = 1 + k // (Ny-2)
        j = 1 + k % (Ny-2)

        A[k, k] = 1 + 2*rx + 2*ry + 1j*Dt/2 * v[i, j]
        M[k, k] = 1 - 2*rx - 2*ry - 1j*Dt/2 * v[i, j]

        if i != 1:
            A[k, (i-2)*(Ny-2)+j-1] = -ry 
            M[k, (i-2)*(Ny-2)+j-1] = ry
            
        if i != Nx-2:
            A[k, i*(Ny-2)+j-1] = -ry
            M[k, i*(Ny-2)+j-1] = ry
        
        if j != 1:
            A[k, k-1] = -rx 
            M[k, k-1] = rx 

        if j != Ny-2:
            A[k, k+1] = -rx
            M[k, k+1] = rx

    return csc_matrix(A), csc_matrix(M)

####################################################

def solveMatrix(A, M, L, Nx, Ny, Ni, Nt, x0, y0, Dy, k):
    """
    Résoudre le système A·x[n+1] = M·x[n] pour chaque pas de temps.

    Args :
        A (ndarray) : Matrice à résoudre.
        M (ndarray) : Matrice à résoudre.
        L (int) : Longueur du domaine.
        Nx (int) : Grandeur du grillage en x.
        Ny (int) : Grandeur du grillage en y.
        Ni (int) : Nombre de coefficients inconnus.
        Nt (int) : Nombre de pas de temps.
        (x0, y0) : Position initiale.

    Returns :
        mod_psis (array) : Vecteur des fonctions d'onde discrétisées.
    """

    solve = factorized(A)  # Pré-factorisation de A pour accélérer les résolutions

    x = np.linspace(0, L, Ny-2)
    y = np.linspace(0, L, Ny-2)
    x, y = np.meshgrid(x, y)
    
    sigma = 1

    psi = psi0(x, y, x0, y0, sigma, k)
    psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0  
    mod_psis = [np.abs(psi)]  
    norms = []

    initial_norm = np.sum(np.abs(psi)**2) * Dy * Dy
    norms.append(initial_norm)

    for i in range(1, Nt):
        psi_vect = psi.reshape(Ni)
        b = M @ psi_vect  # Utilisation directe de l'opérateur sparse
        psi_vect = solve(b)  # Résolution plus rapide
        psi = psi_vect.reshape((Nx-2, Ny-2))

        # Calcul de la norme et vérification de la stabilité
        norme = np.sum(np.abs(psi)**2) * Dy * Dy  # Norme approchée
        norms.append(norme)
        max_psi_at_x0 = np.max(np.abs(psi[:, 0]))  # Valeur max à x=0
        #print(f"Step {i}: Norme = {norme}, max |psi| at x=0: {max_psi_at_x0}")
        
        # Arrêt si la norme explose
        if norme > 1e10:
            print(f"Simulation stopped at step {i}: Norme exploded to {norme}")
            break

        mod_psis.append(np.abs(psi)) 

    return mod_psis, initial_norm, norms

def solveMatrixForConvergence(A, M, L, Nx, Ny, Ni, Nt, x0, y0, Dy, k):
    solve = factorized(A)

    x = np.linspace(0, L, Ny-2)
    y = np.linspace(0, L, Ny-2)
    x, y = np.meshgrid(x, y)
    
    sigma = 1

    psi = psi0(x, y, x0, y0, sigma, k)
    psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0  
    norms = []

    initial_norm = np.sum(np.abs(psi)**2) * Dy * Dy
    norms.append(initial_norm)

    for i in range(1, Nt):
        psi_vect = psi.reshape(Ni)
        b = M @ psi_vect
        psi_vect = solve(b)
        psi = psi_vect.reshape((Nx-2, Ny-2))

        norme = np.sum(np.abs(psi)**2) * Dy * Dy
        norms.append(norme)
        if norme > 1e10:
            print(f"Simulation stopped at step {i}: Norme exploded to {norme}")
            break

    final_psi = np.abs(psi)
    return final_psi, initial_norm, norms

def convergence_erreur(L, T, x0, y0, k, dy_list):
    print("Calcul de l'erreur de convergence numérique...")
    solutions = []
    grids = []

    for Dy in dy_list:
        print(f"Calcul pour Dy = {Dy}...")
        Nx = int(L / Dy) + 1
        Ny = int(L / Dy) + 1
        Dt = Dy**2 / 4
        Nt = int(T / Dt)
        print(f"Nt = {Nt}")

        j0, j1, i0, i1, i2, i3, v, w, s, a, x_fentes = potentielSlits(Dy, Ny, L, y0)
        print(f"Max potential value for Dy = {Dy}: {np.max(np.abs(v)):.6e}")

        Ni = (Nx - 2) * (Ny - 2)
        A, M = buildMatrix(Ni, Nx, Ny, Dy, Dt, v)
        final_psi, initial_norm, norms = solveMatrixForConvergence(A, M, L, Nx, Ny, Ni, Nt, x0, y0, Dy, k)

        final_norm = np.sum(np.abs(final_psi)**2) * Dy * Dy
        print(f"Norme initiale pour Dy = {Dy} : {initial_norm:.6f}")
        print(f"Norme finale pour Dy = {Dy} : {final_norm:.6f}")

        solutions.append(final_psi)
        grids.append((Nx, Ny, Dy))

    errors_l2 = []
    for i in range(len(solutions) - 1):
        psi1 = solutions[i]
        psi2 = solutions[i + 1]
        Nx1, Ny1, Dy1 = grids[i]
        Nx2, Ny2, Dy2 = grids[i + 1]

        factor = int(Dy1 / Dy2)
        target_size = Nx1 - 2
        indices = np.arange(0, Nx2-2, factor)[:target_size]
        psi2_subsampled = psi2[indices, :][:, indices]

        if psi2_subsampled.shape != psi1.shape:
            raise ValueError(f"Dimensions mismatch: psi1 {psi1.shape}, psi2_subsampled {psi2_subsampled.shape}")

        diff = np.abs(psi1 - psi2_subsampled)
        error_l2 = np.sqrt(np.sum(diff**2) * Dy1 * Dy1)
        errors_l2.append(error_l2)

    orders_l2 = []
    for i in range(len(errors_l2) - 1):
        if errors_l2[i + 1] > 0 and errors_l2[i] > 0:
            order = np.log(errors_l2[i] / errors_l2[i + 1]) / np.log(2)
            orders_l2.append(order)
        else:
            orders_l2.append(0)

    print("\nErreurs L^2 (norme euclidienne) :")
    for i, error in enumerate(errors_l2):
        print(f"Dy = {dy_list[i]} -> Dy = {dy_list[i+1]} : Erreur = {error:.6e}")

    print("\nOrdres de convergence (L^2) :")
    for i, order in enumerate(orders_l2):
        print(f"Dy = {dy_list[i+1]} -> Dy = {dy_list[i+2]} : Ordre = {order:.2f}")

    plt.figure(figsize=(8, 6))
    plt.loglog(dy_list[:-1], errors_l2, '-o', label='Erreur L^2')
    dy_ref = np.array(dy_list[:-1])
    error_ref = errors_l2[0] * (dy_ref / dy_ref[0])**2
    plt.loglog(dy_ref, error_ref, '--', label='Pente théorique (ordre 2)')
    plt.xlabel('Pas de discrétisation $Dy$')
    plt.ylabel('Erreur $L^2$')
    plt.title('Convergence numérique (norme euclidienne)')
    plt.grid(True, which="both", ls="--")
    plt.legend()

    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "convergence_error.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

def theoreticalIntensity(y, s, a, D, k, I_0=1):
    """
    Calcule le patron théorique de diffraction en utilisant sin(theta) pour l'angle relatif.

    Args:
        y (array): Coordonnées verticales sur l'écran, centrées autour de L/2.
        s (float): Distance entre les fentes.
        a (float): Largeur effective des fentes.
        D (float): Distance entre le plan des fentes et l'écran.
        k (float): Vecteur d'onde.
        I_0 (float): Intensité maximale (défaut = 1).

    Returns:
        array: Intensité théorique.
    """

    lambda_ = 2 * np.pi / k
    

    y_centered = y 

    # tanθ= (y−L/2)/D

    sin_theta = y_centered / np.sqrt(y_centered**2 + D**2)
    
    # Terme de diffraction (enveloppe)
    sinc_term = np.sinc((np.pi * a * sin_theta) / lambda_)  # np.sinc inclut déjà pi
    
    # Terme d'interférence
    cos_term = np.cos((np.pi * s * sin_theta) / lambda_)
    
    return I_0 * (cos_term**2) * (sinc_term**2)