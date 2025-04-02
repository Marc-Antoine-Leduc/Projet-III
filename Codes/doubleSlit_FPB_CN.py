import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized
from scipy.sparse import lil_matrix, diags

def psi0(x, y, x0, y0, sigma=1, k=15*np.pi):
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

def solveMatrix(A, M, L, Nx, Ny, Ni, Nt, x0, y0, Dy):
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
    
    psi = psi0(x, y, x0, y0)
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

################################################################################
# Ancienne fonctions qui prenaient beaucoup, beaucoup de mémoire et de temps...#
################################################################################

# def buildMatrix(Ni, Nx, Ny, Dy, Dt, v):
#     """
#     Construit la matrice de système d'équations pour la méthode de 
#     Cranck-Nicholson.

#     Args :
#         Ni (int) : Nombre de coefficients inconnus.
#         Nx (int) : Grandeur du grillage en x.
#         Ny (int) : Grandeur du grillage en y.
#         Dy (float) : Pas en y.
#         Dt (float) : Pas de temps.
#         v (float) : Potentiel.

#     Returns :
#         A (ndarray) : Matrice à résoudre.
#         M (ndarray) : Matrice à résoudre.
#     """
#     # Matrices for the Crank-Nicolson calculus. The problem A·x[n+1] = b = M·x[n] will be solved at each time step.
#     A = np.zeros((Ni,Ni), complex)
#     M = np.zeros((Ni,Ni), complex)

#     rx = -Dt/(2j*Dy**2) # Constantes pour simplifier les expression.    
#     ry = -Dt/(2j*Dy**2) # Constante pour simplifier les expressions.

#     # We fill the A and M matrices.
#     for k in range(Ni):     
        
#         # k = (i-1)*(Ny-2) + (j-1)
#         i = 1 + k//(Ny-2)
#         j = 1 + k%(Ny-2)
        
#         # Main central diagonal.
#         A[k,k] = 1 + 2*rx + 2*ry + 1j*Dt/2*v[i,j]
#         M[k,k] = 1 - 2*rx - 2*ry - 1j*Dt/2*v[i,j]
        
#         if i != 1: # Lower lone diagonal.
#             A[k,(i-2)*(Ny-2)+j-1] = -ry 
#             M[k,(i-2)*(Ny-2)+j-1] = ry
            
#         if i != Nx-2: # Upper lone diagonal.
#             A[k,i*(Ny-2)+j-1] = -ry
#             M[k,i*(Ny-2)+j-1] = ry
        
#         if j != 1: # Lower main diagonal.
#             A[k,k-1] = -rx 
#             M[k,k-1] = rx 

#         if j != Ny-2: # Upper main diagonal.
#             A[k,k+1] = -rx
#             M[k,k+1] = rx
#     return A, M
# ####################################################

# def solveMatrix(A, M, L, Nx, Ny, Ni, Nt, x0, y0):
#     """
#     Résoudre le système A·x[n+1] = M·x[n] pour chaque pas de temps.

#     Args :
#         A (ndarray) : Matrice à résoudre.
#         M (ndarray) : Matrice à résoudre.
#         L (int) : Longueur du domaine.
#         Nx (int) : Grandeur du grillage en x.
#         Ny (int) : Grandeur du grillage en y.
#         Ni (int) : Nombre de coefficients inconnus.
#         Nt (int) : Nombre de pas de temps.
#         (x0, y0) : Position initiale.

#     Returns :
#         mod_psis (array) : Vecteur des fonctions d'onde discrétisées.
#     """
#     Asp = csc_matrix(A)

#     x = np.linspace(0, L, Ny-2) # Array of spatial points.
#     y = np.linspace(0, L, Ny-2) # Array of spatial points.
#     x, y = np.meshgrid(x, y)
#     psis = [] # To store the wave function at each time step.

#     psi = psi0(x, y, x0, y0) # We initialise the wave function with the Gaussian.
#     psi[0,:] = psi[-1,:] = psi[:,0] = psi[:,-1] = 0 # The wave function equals 0 at the edges of the simulation box (infinite potential well).
#     psis.append(np.copy(psi)) # We store the wave function of this time step.

#     # We solve the matrix system at each time step in order to obtain the wave function.
#     for i in range(1,Nt):
#         psi_vect = psi.reshape((Ni)) # We adjust the shape of the array to generate the matrix b of independent terms.
#         b = np.matmul(M,psi_vect) # We calculate the array of independent terms.
#         psi_vect = spsolve(Asp,b) # Resolvemos el sistema para este paso temporal.
#         psi = psi_vect.reshape((Nx-2,Ny-2)) # Recuperamos la forma del array de la función de onda.
#         psis.append(np.copy(psi)) # Save the result.

#     # We calculate the modulus of the wave function at each time step.
#     mod_psis = [] # For storing the modulus of the wave function at each time step.
#     for wavefunc in psis:
#         re = np.real(wavefunc) # Real part.
#         im = np.imag(wavefunc) # Imaginary part.
#         mod = np.sqrt(re**2 + im**2) # We calculate the modulus.
#         mod_psis.append(mod) # We save the calculated modulus.
        
#     ## In case there is a need to save memory.
#     # del psis
#     # del M
#     # del psi_vect
#     # del A
#     # del Asp
#     # del b
#     # del im 
#     # del re
#     # del psi

#     return mod_psis

def theoreticalIntensity(y, s, a, D, k, I_0=1):
    """
    Calcule le patron théorique de diffraction en prenant en compte la distance D
    entre les fentes et l'écran d'observation.

    Args:
        y (array) : Coordonnées verticales sur l'écran, centrées autour de 0.
        s (float) : Distance entre les fentes.
        a (float) : Largeur effective des fentes.
        D (float) : Distance entre le plan des fentes et l'écran.
        k (float) : Vecteur d'onde.

    Returns:
        array : Intensité théorique.
    """
    lambda_ = 2 * np.pi / k
    
    sinc_term = np.sinc((a * y) / (lambda_ * D))  # np.sinc inclut déjà pi
    cos_term = np.cos((np.pi * s * y) / (lambda_ * D))
    return I_0 * (cos_term**2) * (sinc_term**2)