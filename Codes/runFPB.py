from potentiel import *
from doubleSlit_FPB_CN import *
from createAnimations import *


if __name__ == "__main__":
    L = 8 # Grandeur de la simulation (de la boîte).
    Dy = 0.05 # Pas d'espace.
    Dt = Dy**2/4 # Pas de temps.
    Nx = int(L/Dy) + 1 # Grandeur du grillage en x.
    Ny = int(L/Dy) + 1 # Grandeur du grillage en y.
    Nt = 500 # Nombre de points de temps.
    v = np.zeros((Ny,Ny), complex)  # Première définition du potentiel.

    # Position initial du faisceau d'électrons.
    x0 = L/5
    y0 = L/2
        
    Ni = (Nx-2)*(Ny-2)  # Nombre d'inconnus v[i,j], i = 1,...,Nx-2, j = 1,...,Ny-2

    j0, j1, i0, i1, i2, i3, v, w = potentielSlits(Dy, Ny, L)

    A, M = buildMatrix(Ni, Nx, Ny, Dy, Dt, v)
    mod_psis = solveMatrix(A, M, L, Nx, Ny, Ni, Nt, x0, y0)

    # animation = makeBasicAnimation(mod_psis, Nt, L)
    animation = makeAnimationForSlits(mod_psis, j0, i0, i1, i2, i3, Dy, Nt, w, L)
