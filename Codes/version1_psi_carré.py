# import numpy as np
# import scipy.sparse as sp
# import scipy.sparse.linalg as spla
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# # Paramètres physiques et numériques
# Lx, Ly = 10.0, 10.0
# Nx, Ny = 100, 100
# dx, dy = Lx / Nx, Ly / Ny
# dt = 0.002
# hbar = 1.0
# m = 1.0

# x = np.linspace(-Lx/2, Lx/2, Nx)
# y = np.linspace(-Ly/2, Ly/2, Ny)
# X, Y = np.meshgrid(x, y)

# # Potentiel cristallin
# V = 5.0 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))

# # Paramètres du faisceau
# omega = 1.5
# k0 = 5.0
# A = 1.0
# r_c = 0.0
# r = np.sqrt(X**2 + Y**2)
# psi_real = A * np.exp(-((r - r_c) ** 2) / omega**2) * np.cos(k0 * r)
# psi_imag = A * np.exp(-((r - r_c) ** 2) / omega**2) * np.sin(k0 * r)
# psi = psi_real + 1j * psi_imag

# # Construction du Laplacien en matrice creuse
# def laplacian_2D(Nx, Ny, dx, dy):
#     Ix = sp.eye(Nx)
#     Iy = sp.eye(Ny)
#     Dx = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(Nx, Nx)) / dx**2
#     Dy = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(Ny, Ny)) / dy**2
#     return sp.kron(Iy, Dx) + sp.kron(Dy, Ix)

# L = laplacian_2D(Nx, Ny, dx, dy)

# # Matrice du système : (I + i dt H)
# I = sp.eye(Nx * Ny, format="csr")
# H = - (hbar / (2 * m)) * L + sp.diags(V.ravel(), 0)  # Hamiltonien total
# A = (I + 1j * dt * H).tocsc()  # Conversion en matrice creuse optimisée
# solver = spla.factorized(A)  # Pré-factorisation LU pour accélérer

# # Animation
# fig, ax = plt.subplots()
# im = ax.imshow(np.abs(psi)**2, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], cmap="inferno", origin="lower")

# def update(frame):
#     global psi
#     psi = solver(psi.ravel()).reshape((Nx, Ny))  # Résolution implicite
#     im.set_data(np.abs(psi)**2)
#     return [im]

# ani = animation.FuncAnimation(fig, update, frames=200, interval=30, blit=False)
# plt.colorbar(im)
# plt.show()

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # Importation pour le graphique 3D

# Paramètres physiques et numériques
Lx, Ly, Lz = 10.0, 10.0, 10.0
Nx, Ny, Nz = 50, 50, 50  # Résolution 3D (augmente la charge de calcul)
dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
dt = 0.002
hbar = 1.0
m = 1.0

x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
z = np.linspace(-Lz/2, Lz/2, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Potentiel cristallin 3D
V = 5.0 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y) + np.cos(2 * np.pi * Z))

# Paramètres du faisceau
omega = 1.5
k0 = 5.0
A = 1.0
r_c = 0.0
r = np.sqrt(X**2 + Y**2 + Z**2)
psi_real = A * np.exp(-((r - r_c) ** 2) / omega**2) * np.cos(k0 * r)
psi_imag = A * np.exp(-((r - r_c) ** 2) / omega**2) * np.sin(k0 * r)
psi = psi_real + 1j * psi_imag

# Construction du Laplacien en matrice creuse 3D
def laplacian_3D(Nx, Ny, Nz, dx, dy, dz):
    Ix = sp.eye(Nx)
    Iy = sp.eye(Ny)
    Iz = sp.eye(Nz)
    Dx = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(Nx, Nx)) / dx**2
    Dy = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(Ny, Ny)) / dy**2
    Dz = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(Nz, Nz)) / dz**2
    return sp.kron(Iz, sp.kron(Iy, Dx)) + sp.kron(Iz, sp.kron(Dy, Ix)) + sp.kron(sp.kron(Iz, Iy), Dz)

L = laplacian_3D(Nx, Ny, Nz, dx, dy, dz)

# Matrice du système
I = sp.eye(Nx * Ny * Nz, format="csr")
H = - (hbar / (2 * m)) * L + sp.diags(V.ravel(), 0)  # Hamiltonien total
A = (I + 1j * dt * H).tocsc()
solver = spla.factorized(A)

# Animation 3D : affichage de la fonction d'onde dans un graphique 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Définir un plan pour z = 0
Z_slice = Nz // 2
X_s, Y_s = np.meshgrid(x, y)
Z_s = np.abs(psi[:, :, Z_slice])**2  # Magnitude de la fonction d'onde

# Tracer la surface 3D
surf = ax.plot_surface(X_s, Y_s, Z_s, cmap="inferno", edgecolor="none")

# Fonction de mise à jour pour l'animation
def update(frame):
    global psi
    psi = solver(psi.ravel()).reshape((Nx, Ny, Nz))  # Résolution implicite
    Z_s = np.abs(psi[:, :, Z_slice])**2  # Met à jour la valeur de la fonction d'onde
    ax.clear()  # Efface l'ancienne surface
    surf = ax.plot_surface(X_s, Y_s, Z_s, cmap="inferno", edgecolor="none")  # Nouvelle surface
    return [surf]

ani = animation.FuncAnimation(fig, update, frames=200, interval=30, blit=False)

# Affichage de la barre de couleur
fig.colorbar(surf)

plt.show()
