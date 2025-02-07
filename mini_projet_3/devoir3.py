L = 0.3     # longueur du mur en mètres
Ti = 293    # Température intérieure en kelvin
Ta = 263    # Température ambiante extérieure en kelvin
q = 2000    # Taux volumique d’émission de la chaleur (W/m^3)
k = 1
h = 1

# Définir le domaine de simulation
N = 100     # Nombre de points de discrétisation
dx = L / N-1


# Conditions frontières
c1 = -k
c2 = h
c3 = -h*Ta
d1 = 0
d2 = 1
d3 = -Ti

# x et t -> np.linspace
# u_n^i -> mp.array


