# PHS3903 - Projet de simulation
# Mini-devoir 1

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import stats

# Paramètres physiques du problème
g = 9.81     # Champ gravitationnel (m²/s)
m = 1.000    # Masse du pendule (kg)
L = 1.000    # Longueur du câble (m)
beta = 0.04  # Constante d'amortissement (1/s)

# Conditions initiales
theta0 = 30         # Position initiale (rad)
omega0 = 1          # Vitesse inititale (rad/s)

# Paramètres généraux de simulation
tf = 1             # Temps final (s)
dt0 = 1.000        # Pas de temps le plus élevé (s)

# Boucle sur le nombre de simulations
K = 5                                       # Nombre de simulations
dt_val = [1, 2, 3, 4, 5]                    # Vecteur des pas de temps pour chaque simulation
thetaf = np.zeros(K)                        # Vecteur des positions finales pour chaque simulation

for k in range(0,K):
# Paramètres spécifiques de la simulation
    dt = dt_val[k]               # Pas de temps de la simulation
    N = 100  # Nombre d'itérations (conseil : s'assurer que dt soit un multiple entier de tf)

# Initialisation
    t = np.arange(0, tf + dt, dt)  # Vecteur des valeurs t_n
    theta = np.zeros(N + 1)  # Vecteur des valeurs theta_n
    theta[0] = 0
    theta[1] = 0

# Exécution
    for n in range(2, N + 1):
        theta[n] = theta[n-1]

    thetaf[k] = theta[-1]  # Position au temps final tf
