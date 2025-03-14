import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

def psi0(x, y, x0, y0, sigma=0.5, k=15*np.pi):
    """
    Fonction d'onde à t=0 (condition initiale).

    Args :
        sigma (float) : Écart-type (défini à 0.5).
        k (complex) : Nombre d'onde. Proportionnel à la quantité de mouvement (15*np.pi).
        (x0, y0) (float) : Positions initiales.
        (x, y) (float) : Système de coordonnées.
        
    Note : 
        si Dy=0.1, utiliser np.exp(-1j*k*(x-x0)), si Dy=0.05, utiliser 
        np.exp(1j*k*(x-x0)) afin que la particule se déplace vers la droite.
    
    Returns :
        conditionInitiale (complex) : Fonction d'onde à t=0.
    """
    conditionInitiale = np.exp(-1/2*((x-x0)**2 + (y-y0)**2)/sigma**2)*np.exp(1j*k*(x-x0))
    return conditionInitiale
####################################################

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
    # Matrices for the Crank-Nicolson calculus. The problem A·x[n+1] = b = M·x[n] will be solved at each time step.
    A = np.zeros((Ni,Ni), complex)
    M = np.zeros((Ni,Ni), complex)

    rx = -Dt/(2j*Dy**2) # Constantes pour simplifier les expression.    
    ry = -Dt/(2j*Dy**2) # Constante pour simplifier les expressions.

    # We fill the A and M matrices.
    for k in range(Ni):     
        
        # k = (i-1)*(Ny-2) + (j-1)
        i = 1 + k//(Ny-2)
        j = 1 + k%(Ny-2)
        
        # Main central diagonal.
        A[k,k] = 1 + 2*rx + 2*ry + 1j*Dt/2*v[i,j]
        M[k,k] = 1 - 2*rx - 2*ry - 1j*Dt/2*v[i,j]
        
        if i != 1: # Lower lone diagonal.
            A[k,(i-2)*(Ny-2)+j-1] = -ry 
            M[k,(i-2)*(Ny-2)+j-1] = ry
            
        if i != Nx-2: # Upper lone diagonal.
            A[k,i*(Ny-2)+j-1] = -ry
            M[k,i*(Ny-2)+j-1] = ry
        
        if j != 1: # Lower main diagonal.
            A[k,k-1] = -rx 
            M[k,k-1] = rx 

        if j != Ny-2: # Upper main diagonal.
            A[k,k+1] = -rx
            M[k,k+1] = rx
    return A, M
####################################################

def solveMatrix(A, M, L, Nx, Ny, Ni, Nt, x0, y0):
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
    Asp = csc_matrix(A)

    x = np.linspace(0, L, Ny-2) # Array of spatial points.
    y = np.linspace(0, L, Ny-2) # Array of spatial points.
    x, y = np.meshgrid(x, y)
    psis = [] # To store the wave function at each time step.

    psi = psi0(x, y, x0, y0) # We initialise the wave function with the Gaussian.
    psi[0,:] = psi[-1,:] = psi[:,0] = psi[:,-1] = 0 # The wave function equals 0 at the edges of the simulation box (infinite potential well).
    psis.append(np.copy(psi)) # We store the wave function of this time step.

    # We solve the matrix system at each time step in order to obtain the wave function.
    for i in range(1,Nt):
        psi_vect = psi.reshape((Ni)) # We adjust the shape of the array to generate the matrix b of independent terms.
        b = np.matmul(M,psi_vect) # We calculate the array of independent terms.
        psi_vect = spsolve(Asp,b) # Resolvemos el sistema para este paso temporal.
        psi = psi_vect.reshape((Nx-2,Ny-2)) # Recuperamos la forma del array de la función de onda.
        psis.append(np.copy(psi)) # Save the result.

    # We calculate the modulus of the wave function at each time step.
    mod_psis = [] # For storing the modulus of the wave function at each time step.
    for wavefunc in psis:
        re = np.real(wavefunc) # Real part.
        im = np.imag(wavefunc) # Imaginary part.
        mod = np.sqrt(re**2 + im**2) # We calculate the modulus.
        mod_psis.append(mod) # We save the calculated modulus.
        
    ## In case there is a need to save memory.
    # del psis
    # del M
    # del psi_vect
    # del A
    # del Asp
    # del b
    # del im 
    # del re
    # del psi

    return mod_psis

