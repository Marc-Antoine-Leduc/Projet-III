# %% [markdown]
# \begin{center}
#     \Huge \textbf{\textcolor{green}{Mini-projet 3 - Matrice 1D}} \\[0.5cm]
#     \Large \textbf{\textcolor{green}{PHS3903}} \\[0.5cm]
#     \Large Alexis Dubé-Valade, Raphäel Villeneuve et Nathan Ferga \\[0.5cm]
#     \large 2212705, 2216147, 2144665 \\[0.5cm]
#     \large 2/12/2025 \\[0.5cm]
#     \large Sean Molesky, Jérémie Villeneuve
# \end{center}

# %% [markdown]
# # Travail préparatoire

# %%
def matrice1D(profil_init, S, x, t,cv, rho, k, xi, c, d):
    alpha = (cv * rho) / k
    
    #Profils initiaux
    N_x = len(x)
    dx = x[1] - x[0]
    
    dt = t[1]-t[0]

    #Init matrice A indep du temps
    A = np.zeros((N_x, N_x))
    A[np.arange(0, N_x), np.arange(0, N_x)] = -2
    A[np.arange(1, N_x), np.arange(0, N_x - 1)] = 1
    A[np.arange(0, N_x - 1), np.arange(1, N_x)] = 1
    
    A[0, [0, 1, 2]] = [2 * c[1] * dx - 3 * c[0], 4 * c[0], -c[0]]
    A[-1, [-3, -2, -1]] = [d[0], -4 * d[0], 2 * d[1] * dx + 3 * d[0]]
    
    #Init vecteur b indep du temps
    b = -S(x)/k * dx**2
    b[0] = -2 * c[2] * dx
    b[-1] = -2 * d[2] * dx
    
    #Init A et b dependant du temps
    M = np.identity(N_x)
    M[0, 0] = 0
    M[-1, -1] = 0
    
    temp = np.zeros((len(t),N_x))
    temp[0,:] = profil_init(x)
    
    b_prime = (M+dt/(alpha*dx**2)*A*(1-xi)) @ temp[0,:] - (dt / (alpha * dx**2)) * b
    A_prime = M - (dt / (alpha * dx**2)) * A*xi
    
    # Calcul de la solution pour chaque pas de temps
    for i in range(1,len(t)):
        temp[i,:] = np.linalg.solve(A_prime, b_prime)
        b_prime = M @ temp[i,:] - (dt / (alpha * dx**2)) * b 
    return temp

# %% [markdown]
# #  

# %% [markdown]
# ## i) Obtention de $T^{max}_{eq}$

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# Équation différentielle: d^2 u/dx^2=g(x) sur x=(a,b)
# Conditions aux limites générales:
# x=a: c1*du/dx+c2*u+c3=0
# x=b: d1*du/dx+d2*u+d3=0

# Équation de transfert de chaleur d^2 T/dx^2=-S(x)/k sur x=(0,L)
# dans un mur d'isolation thermique
L = 0.3; #[m] ; Épaisseur du mur

#k=0.85;h=20; #Valeurs pour vérifier
k=1; h=1
# k=1;#[W/(m*K)]; La conductivité thermique de la brique
# h=1; #[W/(m^2*K)]; Coefficient de transfert thermique pour l'interface plane entre l'air et solide.

# Condition convective (de Robin) à x=0 (face externe du mur): -k*dT/dx=h(Ta-T)
#Température extérieure
Ta=-10; #[oC]
Ta = Ta + 273 #K
#Température intérieure
Ti = 20
Ti = Ti + 273 #K
c1=-k; c2=h; c3=-h*Ta;
# Condition de Dirichler à x=L (face interne du mur): T(L) = Ti, température maintenue constante
d1=0; d2=1; d3=-Ti;

#(N+1) nœuds dans la maille
# Nmax=10000 pour 1G de mémoire


dx = 3/1000 # 3 mm
N = int(L/dx) #Pas de discrétisation
x=np.linspace(0,L,N+1);

S=np.zeros(N+1,dtype=np.double);
A=np.zeros((N+1,N+1),dtype=np.double);
b=np.zeros(N+1,dtype=np.double);
u=np.zeros(N+1,dtype=np.double);

# Sourse volumique de chaleur q[W/m^3] d'épaisseur dL
# La source est intégrée dans la partie intérieure du mur
dL=0.05; 
q=2000; # W/m^3;
#S=q*np.exp(-((x-L)/dL)**2)
S = q/(1 + ((x-L)/dL)**2) 


# matrice pleine
A=np.diag(-2*np.ones(N+1),0)+np.diag(np.ones(N),-1)+np.diag(np.ones(N),1);

#Ajouts des CF dans A
A[0,0]=2*c2*dx-3*c1;
A[0,1]=4*c1;
A[0,2]=-c1;
A[N,N]=3*d1+2*d2*dx;
A[N,N-1]=-4*d1;
A[N,N-2]=d1;

#Construction de b en incluant les CF
b=-S/k*dx**2; 
b[0]=-2*c3*dx; 
b[N]=-2*d3*dx;

#Résolution du système
u=np.linalg.solve(A, b) - 273; # -273 pour reconvertir en oC

Tmax = np.max(u)

plt.figure(figsize=(8,6))
plt.plot(x, u)
# Add a text box in the upper right corner
plt.text(0.7, 0.30, f"T max à l'éq: {Tmax:.3f} $^o$C", 
         fontsize=12, ha="right", va="top", transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
plt.title("Distribution de la température à travers le mur à l'équilibre thermique")
plt.xlabel("Position x (m)")
plt.ylabel("$T_{eq}$ [$^o$C]")
plt.xlim((0,0.3))
plt.show()

# %% [markdown]
# ## ii) Application de la méthode en 1)

# %%
dL = 0.05
L = 0.3
q = 2000
Ta = -10
Ti = 20
h = 1
k = 1
Tw = ((Ti * k / L) + Ta * h) / ((k / L) + h)
dx = 0.003
cv = 1000
rho = 2000
x = np.arange(0, L + dx, dx)
t = np.arange(0, 1e6, dx ** 2 * (cv * rho / k))

input = {"profil_init": lambda x: Tw + (Ti - Tw) * x / L,
         "S": lambda x: q / (1 + ((x - L) / dL) ** 2),
         "x": x,
         "t": t,
         "cv": 1000,
         "rho": 2000,
         "k": 1,
         "xi": 1,
         "c": [-k, h, -h * Ta],
         "d": [0, 1, -Ti]}

temp = matrice1D(**input)   

# %%
fig, ax = plt.subplots()
plt.plot(x,temp[-1,:], label = "$T(x,t_f)$")

ax.set_xlim(0, L)
ax.set_xlabel("Position x (m)")
ax.set_ylabel("Température (°C)")
ax.set_title(f"Évolution de la température - t = {t[-1]:.2g}s")
ax.legend()

plt.show()

# %%
Tmax_array = np.array([np.max(temp[i,:]) for i in range(0,len(t))])
tho_eq = t[Tmax_array > np.max(temp[0])+0.95*(Tmax-np.max(temp[0]))][0]
plt.plot(t, Tmax_array)
plt.plot(t, (np.max(temp[0])+0.95*(Tmax-np.max(temp[0])))*np.ones_like(t))
print(f"Le temps d'équilibrage du système est {tho_eq:.0f}s")

# %% [markdown]
# # Tests

# %%
# Équation différentielle: d^2 u/dx^2=g(x) sur x=(a,b)
# Conditions aux limites générales:
# x=a: c1*du/dx+c2*u+c3=0
# x=b: d1*du/dx+d2*u+d3=0

# Équation de transfert de chaleur d^2 T/dx^2=-S(x)/k sur x=(0,L)
# dans un mur d'isolation thermique
L = 0.3; #[m] ; Épaisseur du mur

#k=0.85;h=20; #Valeurs pour vérifier
k=0.85; h=20
# k=1;#[W/(m*K)]; La conductivité thermique de la brique
# h=1; #[W/(m^2*K)]; Coefficient de transfert thermique pour l'interface plane entre l'air et solide.

# Condition convective (de Robin) à x=0 (face externe du mur): -k*dT/dx=h(Ta-T)
#Température extérieure
Ta=-10; #[oC]
Ta = Ta + 273 #K
#Température intérieure
Ti = 20
Ti = Ti + 273 #K
c1=-k; c2=h; c3=-h*Ta;
# Condition de Dirichler à x=L (face interne du mur): T(L) = Ti, température maintenue constante
d1=0; d2=1; d3=-Ti;

#(N+1) nœuds dans la maille
# Nmax=10000 pour 1G de mémoire


dx = 3/1000 # 3 mm
N = int(L/dx) #Pas de discrétisation
x=np.linspace(0,L,N+1);

S=np.zeros(N+1,dtype=np.double);
A=np.zeros((N+1,N+1),dtype=np.double);
b=np.zeros(N+1,dtype=np.double);
u=np.zeros(N+1,dtype=np.double);

# Sourse volumique de chaleur q[W/m^3] d'épaisseur dL
# La source est intégrée dans la partie intérieure du mur
dL=0.05; 
q=2000; # W/m^3;
#S=q*np.exp(-((x-L)/dL)**2)
S = q/(1 + ((x-L)/dL)**2) 


# matrice pleine
A=np.diag(-2*np.ones(N+1),0)+np.diag(np.ones(N),-1)+np.diag(np.ones(N),1);

#Ajouts des CF dans A
A[0,0]=2*c2*dx-3*c1;
A[0,1]=4*c1;
A[0,2]=-c1;
A[N,N]=3*d1+2*d2*dx;
A[N,N-1]=-4*d1;
A[N,N-2]=d1;

#Construction de b en incluant les CF
b=-S/k*dx**2; 
b[0]=-2*c3*dx; 
b[N]=-2*d3*dx;

#Résolution du système
u=np.linalg.solve(A, b) - 273; # -273 pour reconvertir en oC

Tmax = np.max(u)

plt.figure(figsize=(8,6))
plt.plot(x, u)
# Add a text box in the upper right corner
plt.text(0.7, 0.30, f"T max à l'éq: {Tmax:.3f} $^o$C", 
         fontsize=12, ha="right", va="top", transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
plt.title("Distribution de la température à travers le mur à l'équilibre thermique")
plt.xlabel("Position x (m)")
plt.ylabel("$T_{eq}$ [$^o$C]")
plt.xlim((0,0.3))
plt.show()

# %%
dL = 0.05
L = 0.3
q = 2000
Ta = -10
Ti = 20
h = 20
k = 0.85
Tw = ((Ti * k / L) + Ta * h) / ((k / L) + h)
dx = 0.003
cv = 1000
rho = 2000
x = np.arange(0, L + dx, dx)
t = np.arange(0, 1e6, dx ** 2 * (cv * rho / k))

input = {"profil_init": lambda x: Tw + (Ti - Tw) * x / L,
         "S": lambda x: q / (1 + ((x - L) / dL) ** 2),
         "x": x,
         "t": t,
         "cv": 1000,
         "rho": 2000,
         "k": .85,
         "xi":1,
         "c": [-k, h, -h * Ta],
         "d": [0, 1, -Ti]}

temp = matrice1D(**input)
temp

# %%
fig, ax = plt.subplots()
plt.plot(x, temp[-1, :], label="$T(x,t_f)$")

ax.set_xlim(0, L)
ax.set_xlabel("Position x (m)")
ax.set_ylabel("Température (°C)")
ax.set_title(f"Évolution de la température - t = {t[-1]:.2g}s")
ax.legend()

plt.show()

# %%
Tmax_array = np.array([np.max(temp[i,:]) for i in range(0,len(t))])
tho_eq = t[Tmax_array > np.max(temp[0])+0.95*(Tmax-np.max(temp[0]))][0]
plt.plot(t, Tmax_array)
plt.plot(t, (np.max(temp[0])+0.95*(Tmax-np.max(temp[0])))*np.ones_like(t))
print(f"Le temps d'équilibrage du système est {tho_eq:.0f}s")


