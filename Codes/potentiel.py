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

def potentielSlits(Dy, Ny, L, y0):
    import numpy as np
    
    v0 = 200
    w = 0.2       # épaisseur mur
    s = 0.6       # distance entre centres de fentes
    a = L /(5*L) # np.pi * 2 /k        # hauteur totale de chaque fente
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

    return j0, j1, i0, i1, i2, i3, v, w, s, a, x_center

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