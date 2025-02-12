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

Tn = Ta + (Ti - Ta)*(x/L) 
# 
for n in range(nmax):
    A = np.zeros((N, N))
    b = np.zeros(N)
    
    # --- Condition de Robin à x = 0 (face extérieure) ---
    # Approximons T'(0) par : (-3T0 + 4T1 - T2)/(2*dx)
    # Robin: -k*T'(0) = h0*(Ta - T0)
    # => k*(3T0 - 4T1 + T2) = 2h0*dx*(Ta - T0)
    # => (3k + 2h0*dx)*T0 - 4k*T1 + k*T2 = 2h0*dx*Ta
    A[0, 0] = 3*k + 2*h0*dx
    A[0, 1] = -4*k
    A[0, 2] = k
    b[0] = 2*h0*dx*Ta

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
