import numpy as np

def potentiel(y, mu=0, sigma=1):
    """
    Calcule le potentiel gaussien unique selon l'équation V(y) = (1/(σ√2π)) * exp(-(y-μ)²/(2σ²))
    
    Paramètres:
        y : array ou float, coordonnée(s) où évaluer le potentiel
        mu : float, position centrale du potentiel (défaut = 0)
        sigma : float, écart-type du potentiel gaussien (défaut = 1)
    
    Retourne:
        float ou array, valeur du potentiel à la position y
    """
    facteur = 1 / (sigma * np.sqrt(2 * np.pi))
    exposant = -((y - mu)**2) / (2 * sigma**2)
    return facteur * np.exp(exposant)

def potentiel_periodique(y, a, sigma=1, L=10):
    """
    Calcule le potentiel périodique total selon l'équation V(y) = Σ [1/(σ√2π)] * exp(-(y-na)²/(2σ²))
    
    Paramètres:
        y : array ou float, coordonnée(s) où évaluer le potentiel
        a : float, paramètre de maille du cristal (distance entre atomes)
        sigma : float, écart-type du potentiel gaussien (défaut = 1)
        L : int, nombre d'atomes de chaque côté (total 2L+1 termes) (défaut = 10)
        
    Retourne:
        float ou array, somme des potentiels à la position y
    """

    V_total = np.zeros_like(y, dtype=float)
    
    for n in range(-L, L + 1):
        V_total += potentiel(y, mu=n*a, sigma=sigma)
    
    return V_total

def potentielPeriodicSlits(Dy, Ny, L, s, a, w, v0):
    """
    Crée un mur vertical avec des fentes périodiques.
    
    - Dy : pas spatial
    - Ny : nombre de points sur y
    - L  : taille du domaine
    - s  : espacement entre les centres des fentes
    - a  : largeur verticale de chaque fente
    - w  : épaisseur du mur (en x)
    - v0 : hauteur du potentiel
    """
    slit_half = a / 2
    j0 = int(round((L / 3 - w/2) / Dy))
    j1 = int(round((L / 3 + w/2) / Dy))
    
    v = v0 * np.ones((Ny, Ny), dtype=complex)
    
    # Coordonnées des centres des fentes
    # slit_centers = np.arange(s/2, L, s)
    slit_centers = np.arange(((L+s)/2) - 10*s,((L+s)/2) + 10*s,s)
    
    for yc in slit_centers:
        i0 = int(round((yc - slit_half) / Dy))
        i1 = int(round((yc + slit_half) / Dy))
        
        # On s'assure qu'on ne sort pas du domaine
        i0 = max(i0, 0)
        i1 = min(i1, Ny)
        
        # Création de la fente (potentiel nul)
        v[i0:i1, j0:j1] = 0

    return j0, j1, v



def potentielGaussianSlits(Dx, Dy, Nx, Ny, L, s, sigma, w, v0):
    """
    Crée un mur vertical avec des fentes périodiques modélisées par des potentiels gaussiens inversés.

    - Dx, Dy : pas spatiaux
    - Nx, Ny : nombre de points (x, y)
    - L      : taille du domaine
    - s      : espacement entre les fentes (dans y)
    - sigma  : largeur (écart-type) des gaussiennes
    - w      : épaisseur du mur (dans x)
    - v0     : hauteur maximale du mur
    """
    # Création du mur plein (barrière haute)
    V = v0 * np.ones((Ny, Nx), dtype=complex)

    # Indices x du mur
    x_wall_center = L / 3
    j0 = int(round((x_wall_center - w/2) / Dx))
    j1 = int(round((x_wall_center + w/2) / Dx))
    
    y_vals = np.linspace(0, L, Ny)
    x_vals = np.linspace(0, L, Nx)

    # Coordonnées des centres des gaussiennes (dans y uniquement)
    slit_centers = np.arange(s / 2, L, s)

    for yc in slit_centers:
        for j in range(j0, j1):  # Pour chaque colonne dans l'épaisseur du mur
            V[:, j] -= v0 * np.exp(-((y_vals - yc) ** 2) / (2 * sigma ** 2))

    # Clamp : on s'assure que V reste ≥ 0
    V = np.clip(V, 0, v0)

    return j0, j1, V, x_wall_center




def potentielSlits(Dy, Ny, L, y0, s, w, v0, a):

    slit_half = a / 2

    # -- Calcul des indices horizontaux (j0, j1) pour positionner le mur au centre en x --
    x_center = L / 3
    j0 = int(round((x_center - w/2)/Dy))
    j1 = int(round((x_center + w/2)/Dy))

    # -- Calcul des indices verticaux (i0..i3) pour avoir 2 fentes symétriques autour de y0 --
    lower_slit_center = y0 - s/2
    upper_slit_center = y0 + s/2
    
    i0 = int(round((lower_slit_center - slit_half)/Dy))
    i1 = int(round((lower_slit_center + slit_half)/Dy))
    i2 = int(round((upper_slit_center - slit_half)/Dy))
    i3 = int(round((upper_slit_center + slit_half)/Dy))
    
    # -- On s'assure qu'on ne sort pas de la grille [0..Ny-1] --
    i0 = max(i0, 0)
    i1 = min(i1, Ny)
    i2 = max(i2, 0)
    i3 = min(i3, Ny)
    
    v = np.zeros((Ny, Ny), dtype=complex)
    
    v[:i0,    j0:j1] = v0
    v[i1:i2,  j0:j1] = v0
    v[i3:,    j0:j1] = v0

    return j0, j1, i0, i1, i2, i3, v, x_center

def showPotential(v, L, title="Visualisation du Potentiel"):
    """
    Affiche le potentiel v sous forme d'image 2D.
    
    Args:
        v (ndarray): Matrice 2D représentant le potentiel, de taille (Ny, Nx).
        L (float)  : Longueur du domaine de simulation en x et y.
        title (str): Titre à afficher sur la figure.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    Ny, Nx = v.shape

    x = np.linspace(0, L, Nx)
    y = np.linspace(0, L, Ny)
    
    plt.figure(figsize=(6, 5))
    
    plt.imshow(v.real, extent=[0, L, 0, L], origin='lower', 
               cmap='hot', aspect='auto')
    
    plt.colorbar(label='Potentiel (Re[v])')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')


def potentiel_absorbant(x, y, L, v, d_abs=0.5, strength=100):
    """
    Essaie pour ne pas avoir de réflexsion.
    """
    # x = np.linspace(0, L, Ny)  # Coordonnées complètes (y compris bords)
    # y = np.linspace(0, L, Ny)
    # x, y = np.meshgrid(x, y)

    v_abs = np.zeros_like(x, dtype=complex)
    mask_right = x > (L - d_abs)
    v_abs[mask_right] = 1j * strength * ((x[mask_right] - (L - d_abs)) / d_abs)**2
    mask_left = x < d_abs
    v_abs[mask_left] = 1j * strength * ((d_abs - x[mask_left]) / d_abs)**2
    mask_top = y > (L - d_abs)
    v_abs[mask_top] += 1j * strength * ((y[mask_top] - (L - d_abs)) / d_abs)**2
    mask_bottom = y < d_abs
    v_abs[mask_bottom] += 1j * strength * ((d_abs - y[mask_bottom]) / d_abs)**2

    plt.imshow(np.abs(v), extent=[0, L, 0, L], origin='lower')
    plt.colorbar(label='|v|')
    plt.title('Potentiel total (fentes + absorbant)')

    return v_abs  # Imaginaire car introduit une perte, analogue à partie imaginaire de la permittivité électrique

if __name__ == "__main__":
    y = np.linspace(-10, 10, 1000)
    
    sigma = 0.5  # écart-type
    a = 5.0     # paramètre de maille
    L = 5       # nombre d'atomes de chaque côté
    
    V_single = potentiel(y, mu=0, sigma=sigma)
    V_periodic = potentiel_periodique(y, a=a, sigma=sigma, L=L)
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(y, V_single)
    plt.title("Potentiel Gaussien Unique")
    plt.xlabel("y")
    plt.ylabel("V(y)")
    
    plt.subplot(1, 2, 2)
    plt.plot(y, V_periodic)
    plt.title("Potentiel Périodique")
    plt.xlabel("y")
    plt.ylabel("V(y)")
    
    plt.tight_layout()
    plt.show()