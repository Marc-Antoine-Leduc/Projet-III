import numpy as np
import matplotlib.pyplot as plt

L    = 0.3       # [m] épaisseur du mur
k    = 1.0       # [W/m.K] conductivité
h0   = 1.0       # [W/m^2.K] coeff. convectif côté x=0
hL   = 1.0       # [W/m^2.K] coeff. convectif côté x=L
Ta   = 263.0     # [K] côté x=0
Ti   = 293.0     # [K] côté x=L

rho  = 1.0       # [kg/m^3] 
Cp   = 1.0       # [J/kg.K]  

q    = 2000.0    # [W/m^3] amplitude de la source
dL   = L/10.0   

# Condition initiale linéaire
#    T_w = ( Ti*(k/L) + Ta*h0 ) / ( (k/L) + h0 )

T_w = (Ti*(k/L) + Ta*h0) / ((k/L) + h0)

N   = 101              # nombre de points
dx  = L/(N-1)
x   = np.linspace(0, L, N)

dt   = 0.1
nmax = 200

# Condition initiale: T(x,0) = T_w + (Ti - T_w)*[ x / L ]

Tn = T_w + (Ti - T_w)*(x / L)

# Reformuler équation avec coefficients pour méthode implicite
#    r = alpha * dt / dx^2, où alpha = k / (rho*Cp)

alpha = k/(rho*Cp)
r     = alpha*dt/(dx**2)

for n in range(nmax):

    A = np.zeros((N, N))
    b = np.zeros(N)

    # Nœuds intérieurs (1..N-2)

    #    Équation implicite :
    #    (1 + 2r)*T_i^{p+1} - r*T_{i-1}^{p+1} - r*T_{i+1}^{p+1}
    #       = T_i^p + (dt/(rho*Cp))*S(x_i)

    for i in range(1, N-1):
        A[i, i-1] = -r
        A[i, i]   = 1 + 2*r
        A[i, i+1] = -r

        # Source S(x) = q * [1 + ((x - L)/dL)^2]
        S_i   = q * (1.0 + ((x[i]-L)/dL)**2)
        b[i]  = Tn[i] + dt*(S_i/(rho*Cp))

    # Condition de Robin au bord x=0
    #
    #    -k dT/dx(0) = h0*( Ta - T(0) )
    #    => alpha0 = (h0 * dx)/k
    #    =>  (1 + 2*r*alpha0)*T_0 - 2*r*T_1 = T_0^p + ... + 2*r*alpha0*Ta

    alpha0 = (h0 * dx) / k

    A[0, 0] = 1 + 2*r*alpha0
    A[0, 1] = -2*r

    S_0 = q * (1.0 + ((x[0]-L)/dL)**2)
    b[0] = Tn[0] + dt*(S_0/(rho*Cp)) + 2*r*alpha0*Ta

    # Condition de Robin au bord x=L
    #
    #    -k dT/dx(L) = hL*( Ti - T(L) )
    #    => alphaL = (hL * dx)/k
    #    => (1 + 2*r*alphaL)*T_{N-1} - 2*r*T_{N-2} = ...
    #                        ... = T_{N-1}^p + 2*r*alphaL*Ti + ...

    alphaL = (hL * dx) / k

    A[N-1, N-1] = 1 + 2*r*alphaL
    A[N-1, N-2] = -2*r

    S_L = q * (1.0 + ((x[N-1]-L)/dL)**2)
    b[N-1] = Tn[N-1] + dt*(S_L/(rho*Cp)) + 2*r*alphaL*Ti

    Tnp1 = np.linalg.solve(A, b)

    # Mise à jour du profil de température

    Tn = Tnp1.copy()

plt.figure(figsize=(6,4))
plt.plot(x, Tn, label="T(x) finale")
plt.xlabel("x [m]")
plt.ylabel("Température [K]")
plt.title("Équation de la chaleur 1D")
plt.grid(True)
plt.legend()
plt.show()
