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

    # x = np.linspace(0, L, Ny)  # Coordonnées complètes (y compris bords)
    # y = np.linspace(0, L, Ny)
    # x, y = np.meshgrid(x, y)

    # v_abs = potentiel_absorbant(x, y, L, d_abs=2, strength=100) # Fucking instable
    # v += v_abs

    plt.imshow(np.abs(v), extent=[0, L, 0, L], origin='lower')
    plt.colorbar(label='|v|')
    plt.title('Potentiel total (fentes + absorbant)')
    

    A, M = buildMatrix(Ni, Nx, Ny, Dy, Dt, v)
    mod_psis, initial_norm = solveMatrix(A, M, L, Nx, Ny, Ni, Nt, x0, y0, Dy)

    final_psi = mod_psis[-1]  # Dernière étape temporelle
    screen_intensity = np.abs(final_psi[:, -1])**2  # Intensité (|psi|^2) sur le bord droit
    y_screen = np.linspace(0, L, Ny-2)  # Coordonnées y le long de l’écran

    final_norm = np.sum(np.abs(final_psi)**2) * Dy * Dy

    print(f"Probabilité totale initiale : {initial_norm}")
    print(f"Probabilité totale finale : {final_norm}")
    print(f"Fonction d'onde bien normalisée : {0.95 <= initial_norm <= 1.05 and 0.95 <= final_norm <= 1.05}")

    # Affichage du patron de diffraction
    plt.figure(figsize=(8, 6))
    plt.plot(y_screen, screen_intensity, label='Patron de diffraction')
    plt.xlabel('Position y')
    plt.ylabel('Intensité (|ψ|^2)')
    plt.title('Patron de diffraction sur l’écran à x = L')
    plt.grid(True)
    plt.legend()

    plt.show()

    animation = makeAnimationForSlits(mod_psis, j0, i0, i1, i2, i3, Dy, Nt, w, L)

