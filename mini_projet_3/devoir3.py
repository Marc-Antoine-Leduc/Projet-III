import numpy as np 
import matplotlib.pyplot as plt

# Paramètres physiques
L    = 0.3        # [m] épaisseur du mur
k    = 1.0        # [W/(m·K)] conductivité thermique
h0   = 1.0        # [W/(m²·K)] coefficient de convection à x = 0 (face extérieure)
Ta   = -1.0      # [°C] température ambiante extérieure (x = 0)
Ti   = 20.0       # [°C] température imposée à l'intérieur (x = L)
rho  = 1         # [kg/m³]
Cp   = 1         # [J/(kg·K)]

q    = 2000.0    # [W/m³] amplitude de la source volumique
dL   = L/20.0    # [m] largeur caractéristique de la source

N = 101                     # nombre de nœuds
dx = L/(N-1)                # pas spatial
x = np.linspace(0, L, N)

dt = 0.1                    # pas de temps [s]
nmax = 200                  # nombre de pas de temps

# Coefficient de diffusivité et paramètre r pour le schéma implicite
alpha = k/(rho*Cp)
r = alpha*dt/(dx**2)

T_w = (Ti*(k/L) + Ta*h0) / ((k/L) + h0)

# Condition initiale : T(x,0) = T_w + (T_i - T_w)*(x/L)
Tn = T_w + (Ti - T_w)*(x/L)

# --- Définition des coefficients pour les conditions aux limites ---
# Pour x = 0 (Robin):
# Condition physique : -k T'(0) = h0 (Ta - T(0))
# Écriture sous forme : c1*T'(0) + c2*T(0) + c3 = 0
c1 = -k
c2 = h0
c3 = -h0*Ta

# Pour x = L (Dirichlet exprimée comme condition de Robin):
# d1*T'(L) + d2*T(L) + d3 = 0, avec T(L)=T_i
d1 = 0
d2 = 1
d3 = -Ti

for n in range(nmax):
    A = np.zeros((N, N))
    b = np.zeros(N)
    
    # --- Condition de Robin à x = 0 (face extérieure) ---
    # Approximons T'(0) par : (-3*T0 + 4*T1 - T2)/(2*dx)
    # On a donc : c1*(-3*T0+4*T1-T2)/(2*dx) + c2*T0 + c3 = 0
    # En multipliant par 2*dx et en substituant c1, c2 et c3, on obtient :
    # (3*k + 2*h0*dx)*T0 - 4*k*T1 + k*T2 = 2*h0*dx*Ta
    A[0, 0] = 3*k + 2*h0*dx
    A[0, 1] = -4*k
    A[0, 2] = k
    b[0] = 2*h0*dx*Ta

    for i in range(1, N-1):
        A[i, i-1] = -r
        A[i, i]   = 1 + 2*r
        A[i, i+1] = -r
        
        # Terme source : S(x) = q / [1 + ((x - L)/dL)^2]
        S_i = q / (1.0 + ((x[i]-L)/dL)**2)
        b[i] = Tn[i] + dt*(S_i/(rho*Cp))
    
    # --- Condition de Dirichlet à x = L (face intérieure) ---
    # On impose T(L)=T_i en utilisant la forme d'une condition de Robin avec d1=0, d2=1, d3=-T_i
    A[N-1, :] = 0.0
    A[N-1, N-1] = 1.0
    b[N-1] = Ti

    Tnp1 = np.linalg.solve(A, b)
    Tn = Tnp1.copy()  

plt.figure(figsize=(6,4))
plt.plot(x, Tn, 'b-', label='T(x) finale')
plt.xlabel('x [m]')
plt.ylabel('Température [°C]')
plt.title("Évolution temporelle (méthode implicite avec condition de Robin)")
plt.grid(True)
plt.legend()
plt.show()
