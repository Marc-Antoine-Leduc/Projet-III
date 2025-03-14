from potentiel import *
from doubleSlit_FPB_CN import *
from createAnimations import *


if __name__ == "__main__":
    L = 8 # Well of width L. Shafts from 0 to +L.
    Dy = 0.05 # Spatial step size.
    Dt = Dy**2/4 # Temporal step size.
    Nx = int(L/Dy) + 1 # Number of points on the x axis.
    Ny = int(L/Dy) + 1 # Number of points on the y axis.
    Nt = 500 # Number of time steps.

    # Initial position of the center of the Gaussian wave function.
    x0 = L/5
    y0 = L/2
        
    Ni = (Nx-2)*(Ny-2)  # Number of unknown factors v[i,j], i = 1,...,Nx-2, j = 1,...,Ny-2

    j0, j1, i0, i1, i2, i3, v, w = potentielSlits(Dy, Ny)

    A, M = buildMatrix(Ni, Nx, Ny, Dy, Dt, v)
    mod_psis = solveMatrix(A, M, L, Nx, Ny, Ni, Nt, x0, y0)
    
    animation = makeAnimationForSlits(mod_psis, j0, i0, i1, i2, i3, Dy, Nt, w, L)
    saveData(mod_psis)