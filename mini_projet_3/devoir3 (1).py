import numpy as np
import matplotlib.pyplot as plt

L    = 0.3        # [m] épaisseur du mur
k    = 1.0        # [W/(m·K)] conductivité thermique
h0   = 1.0        # [W/(m²·K)] coefficient de convection à x = 0 (face extérieure)
Ta   = -10.0      # [°C] température ambiante extérieure (x = 0)
Ti   = 20.0       # [°C] température imposée à l'intérieur (x = L)
rho  = 1        # [kg/m³]
Cp   = 1        # [J/(kg·K)]

q    = 2000.0     # [W/m³] amplitude de la source volumique
dL   = L/20.0    # [m] largeur caractéristique de la source

N = 101                     # nombre de nœuds
dx = L/(N-1)                # pas spatial
x = np.linspace(0, L, N)

dt = 0.1                    # pas de temps [s]
nmax = 200                  # nombre de pas de temps

# Coefficient de diffusivité et paramètre r
alpha = k/(rho*Cp)
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
        b[i] = Tn[i] + dt*(S_i/(rho*Cp))
    
    # --- Condition de Dirichlet à x = L (face intérieure) ---
    A[N-1, :] = 0.0
    A[N-1, N-1] = 1.0
    b[N-1] = Ti

    Tnp1 = np.linalg.solve(A, b)
    Tn = Tnp1.copy()  

plt.figure(figsize=(6,4))
plt.plot(x, Tn, 'b-', label='T(x) finale')
plt.xlabel('x [m]')
plt.ylabel('Température [°C]')
plt.title("Évolution temporelle (méthode implicite)")
plt.grid(True)
plt.legend()
plt.show()

