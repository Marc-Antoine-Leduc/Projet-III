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

if __name__ == "__main__":
    y = np.linspace(-10, 10, 1000)
    
    sigma = 0.5  # écart-type
    a = 2.0     # paramètre de maille
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