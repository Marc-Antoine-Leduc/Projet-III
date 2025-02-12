import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve

L    = 0.3        # [m] épaisseur du mur
k    = 1.0        # [W/(m·K)] conductivité thermique
h0   = 1.0        # [W/(m²·K)] coefficient de convection à x = 0 (face extérieure)
Ta   = -10.0      # [°C] température ambiante extérieure (x = 0)
Ti   = 20.0       # [°C] température imposée à l'intérieur (x = L)
rho  = 2000        # [kg/m³]
Cv   = 1000        # [J/(kg·K)]

q    = 2000.0     # [W/m³] amplitude de la source volumique
dL   = L/20.0    # [m] largeur caractéristique de la source

N = 101                     # nombre de nœuds
dx = L/(N-1)                # pas spatial
x = np.linspace(0, L, N)

alpha = k / (rho * Cv)
dt = 1 * dx**2/ alpha                    # pas de temps [s]
nmax = 200                  # nombre de pas de temps

# Coefficient de diffusivité et paramètre r
r = alpha*dt/(dx**2)

# Calcul de Tw selon l'équation donnée
Tw = (Ti * k / L + Ta * h0) / (k / L + h0)

# Initialisation correcte de la température
Tn = Tw + (Ti - Tw) * (x / L)  # Condition initiale exacte


# --- Condition de Robin à x = 0 (face extérieure) ---
c1 = 3*k + 2*h0*dx
c2 = -4*k
c3 = k
b0 = 2*h0*dx*Ta

# --- Condition de Dirichlet à x = L (face intérieure) ---
d1 = 0
d2 = 1
d3 = 0
bN = Ti

# 
for n in range(nmax):
    A = np.zeros((N, N))
    b = np.zeros(N)

    A[0, 0] = c1
    A[0, 1] = c2
    A[0, 2] = c3
    b[0] = b0
    
    # --- Condition de Robin à x = 0 (face extérieure) ---
    # Approximons T'(0) par : (-3T0 + 4T1 - T2)/(2*dx)
    # Robin: -k*T'(0) = h0*(Ta - T0)
    # => k*(3T0 - 4T1 + T2) = 2h0*dx*(Ta - T0)
    # => (3k + 2h0*dx)*T0 - 4k*T1 + k*T2 = 2h0*dx*Ta
    A[N-1, :] = 0.0
    A[N-1, N-1] = d2
    b[N-1] = bN

    # --- Nœuds intérieurs i = 1, ..., N-2 ---
    for i in range(1, N-1):
        A[i, i-1] = -r
        A[i, i]   = 1 + 2*r
        A[i, i+1] = -r
        
        # Terme source S(x) = q / [1 + ((x-L)/dL)^2]
        S_i = q / (1.0 + ((x[i]-L)/dL)**2)
        # Le côté droit intègre la contribution du terme source et la condition temporelle
        b[i] = Tn[i] + dt*(S_i/(rho*Cv))
    
    # --- Condition de Dirichlet à x = L (face intérieure) ---
    A[N-1, :] = 0.0
    A[N-1, N-1] = 1.0
    b[N-1] = Ti

    Tnp1 = np.linalg.solve(A, b)
    Tn = Tnp1.copy()  

T_eq_max = np.max(Tnp1)
# Méthode implicite pour le régime transitoire
dt = 1 * dx**2 / alpha  # Réduction de dt pour assurer la stabilité
time_elapsed = 0
tolerance = 0.05 * (Tnp1 - Ta)
T = Tnp1.copy()  # Initialiser avec la solution stationnaire
B = (np.identity(N) - alpha * dt * A)


# Stockage des valeurs pour le tracé
time_values = []
T_max_values = []

# print(f'Initial: T_max = {np.max(T):.2f}, Target = {T_eq_max - tolerance:.2f}')

# Boucle temporelle
for i in range(len(tolerance)):
    while T[i] < T_eq_max - tolerance[i]:
        T = np.linalg.solve(B, T + ((dt * S_i) / (rho * Cv)))
        time_elapsed += dt
        time_values.append(time_elapsed)
        T_max_values.append(np.max(T))
        
        if len(time_values) > 100000:  # Sécurité pour éviter une boucle infinie
            print("Erreur : la boucle semble ne jamais converger.")
            break

# Vérification et correction des valeurs négatives
if len(T_max_values) == 0:
    print("Erreur : aucune donnée enregistrée pour T_max_values. La boucle ne s'est pas exécutée.")
else:
    T_diff = np.abs(T_max_values - T_eq_max) + 1e-10  # Ajout d'une petite valeur pour éviter les zéros
    
    # Affichage de la courbe |T_max - T_eq_max| en fonction du temps
    plt.figure(figsize=(8, 5))
    plt.plot(time_values, T_diff, label='$|T_{max}(t) - T_{eq}^{max}|$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Temps $t$ [s]')
    plt.ylabel('$|T_{max}(t) - T_{eq}^{max}|$ [°C]')
    plt.title('Évolution de la température maximale')
    plt.legend()
    plt.grid()
    plt.show()

print(f'Température maximale à léquilibre: {T_eq_max:.2f} °C')
print(f'Temps déquilibrage: {time_elapsed:.2f} s')

