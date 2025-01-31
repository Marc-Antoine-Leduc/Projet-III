import numpy as np
import matplotlib.pyplot as plt

# Paramètres physiques du problème
g = 9.81     # Champ gravitationnel (m²/s)
m = 1.000    # Masse du pendule (kg)
L = 1.000    # Longueur du câble (m)
beta = 0.1   # Constante d'amortissement (1/s)
pi = np.pi   # pi

# Conditions initiales
theta0 = pi/6       # Position initiale (rad)
omega0 = 5          # Vitesse inititale (rad/s)

# Paramètres généraux de simulation
tf = 10             # Temps final (s)
dt0 = 0.1           # Pas de temps le plus élevé (s)

# Boucle sur le nombre de simulations
K = 5                                       # Nombre de simulations
dt_val = [dt0, dt0/2, dt0/4, dt0/8, dt0/16] # Vecteur des pas de temps pour chaque simulation
thetaf = np.zeros(K)                        # Vecteur des positions finales pour chaque simulation
position_finale = []                        # Initialiser un vecteur position finale
firstTry = True

for k in range(0,K):
# Paramètres spécifiques de la simulation
    dt = dt_val[k]    # Pas de temps de la simulation
    N = int(tf / dt)  # Nombre d'itérations (conseil : s'assurer que dt soit un multiple entier de tf)

# Initialisation
    t = np.linspace(0, tf, N)
    theta = np.zeros(N)
    theta[0] = theta0
    theta[1] = theta0 + (1 - (beta * dt) / 2) * omega0 * dt - (g * dt**2 / (2 * L)) * np.sin(theta0)

# Exécution
    for n in range(2, len(theta)):
        theta[n] =  (4 * theta[n-1] 
                           - (2 - beta * dt) * theta[n - 2] 
                           - (2 * g * (dt**2)/ L) * np.sin(theta[n-1])) / (2 + beta * dt)
    
    position_finale.append(theta[-1])

    if firstTry:
        plt.figure(figsize=(8, 5))
        plt.plot(t, theta, label=r'$\theta(t)$', color='b')
        plt.xlabel('Temps (s)')
        plt.ylabel('Angle $\Theta$ (rad)')
        plt.title('Évolution de $\Theta(t)$ pour le pendule amorti avec un pas de temps de 0,1s')
        plt.grid()
        plt.show()
        firstTry = False

############################
############################

import pandas as pd
from tabulate import tabulate

data = {
    "Pas (s)": dt_val,
    "Position finale (m)": position_finale,
}

df = pd.DataFrame(data)
titre = "Position du pendule au temps t=10s pour différents pas"

# Affichage avec un titre
print(f"\n{titre}\n" + "=" * len(titre))  # Affichage du titre avec soulignement
print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))