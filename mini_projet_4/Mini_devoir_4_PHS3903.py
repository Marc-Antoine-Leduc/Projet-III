import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# Distribution de la température dans un appartement d'un immeuble aux plusieurs étages


# Équation de transfert de chaleur:
# k*(d^2 T(x,y)/dx^2 + d^2 T(x,y)/dy^2)+S=0


# Conditions aux limites:

# (1) Condition convective (de Robin) à x=0 et à x=Lx (faces externes du mur):
# -k*dT(x=0,y)/dx=-h*(T-Ta)
# -k*dT(x=L,y)/dx=h*(T-Ta)
Ta=-20; #oC

# (2) Condition de Dirichlet sur le plafond et sur le plancher
# T(x, y=0 ou y=Ly)=Tp
Tp=20; #oC

# Dimensions d'appartement
Lx=4; #[m]
Ly=2.4;  #[m]

# Parametres d'un mur d'isolation thermique
Lm=0.4; #m ; Épaisseur du mur en brique
km=0.85;#W/(m*K); La conductivité thermique de la brique
h=10; #W/(m^2*K); Coefficient de transfert thermique sur les surfaces extérieures du mur

# Paramètres de l'air qui remplit l'appartement
ka=0.024;

fact_ar = np.array([1.0, 0.5], dtype=np.double)    # [1.0, 0.5, 0.25, 0.125, 0.0625]
d_ar=np.zeros(fact_ar.size,dtype=np.double);
tini_ar=np.zeros(fact_ar.size,dtype=np.double);
tinv_ar=np.zeros(fact_ar.size,dtype=np.double);
mem_ar=np.zeros(fact_ar.size,dtype=np.double);
Tm_ar=np.zeros(fact_ar.size,dtype=np.double);
Err_ar=np.zeros(fact_ar.size-1,dtype=np.double);
d_Err_ar=np.zeros(fact_ar.size-1,dtype=np.double);

ci=-1;
for fact in fact_ar:
    ci=ci+1;
    d=0.1*fact;    #Pas de discrétisation en [m]
    print('Pas de discritization dx=dy=',d,'m')
    d_ar[ci]=d;
    Nx=int(np.rint(Lx/d+1)); # Nombre de nœuds le long de X
    Ny=int(np.rint(Ly/d+1)); # Nombre de nœuds le long de Y
    
    
    tic=time.time_ns();
    
    # Initialisation de la source de chaleur, de la conductivité thermique et de la matrice
    S=np.zeros((Ny,Nx),dtype=np.double);
    k=np.zeros((Ny,Nx),dtype=np.double);
    for i in np.arange(1,Ny+1,1): #i=1,..,Ny - numérotation des nœuds sur un maillage physique
        y=(i-1)*d;
        for j in np.arange(1,Nx+1,1): #j=1,..,Nx - numérotation des nœuds sur un maillage physique
            x=(j-1)*d;
            
            # Sourse volumique de chaleur q[W/m^3] d'épaisseur dL.
            # La source est intégrée dans les parties intérieures du mur à x=Lm et à x=Lx-Lm et
            # il occupe les tiers du mur dans la direction verticale
            dL=0.1;
            q=2.0e4;# W/m^3;
            if ((x<=Lm) and (y<=Ly/3+Lm) and (y>Lm)):
                # À l'intérieur de l'élément chauffant
                S[i-1,j-1]=q*np.exp(-((x-Lm)/dL)**2);
            elif ((x>=(Lx-Lm)) and (y<=Ly/3+Lm) and (y>Lm)):
                # À l'intérieur de l'élément chauffant
                S[i-1,j-1]=q*np.exp(-((Lx-Lm-x)/dL)**2);
            else:
                # À l'extérieur de l'élément chauffant
                S[i-1,j-1]=0.0;
            
            # L'espace de vie de l'appartement est délimité par
            # les parois d'épaisseur Lm à tous les quatre côtés
            if ((x<=Lm) or (x>=(Lx-Lm)) or (y<=Lm) or (y>=(Ly-Lm))):
                # À l'intérieur du mur
                k[i-1,j-1]=km;
            else:
                # À l'intérieurde de l'appartement
                k[i-1,j-1]=ka;
               
    M = lil_matrix((Nx*Ny, Nx*Ny), dtype=np.double)
    b=np.zeros((Nx*Ny,1),dtype=np.double);
    T=np.zeros((Nx*Ny,1),dtype=np.double);
    Tr=np.zeros((Ny,Nx),dtype=np.double);
    
    for i in np.arange(1,Ny+1,1):
        for j in np.arange(1,Nx+1,1):
            # remplir la ligne pl de la matrice M
            pl=(i-1)*Nx+j;
            
            if (((i>1) and (i<Ny)) and ((j>1) and (j<Nx))):
                # noeud qui est strictement à l'intérieur de la cellule de simulation
                pc=pl;M[pl-1,pc-1]=-4; # contribution de noeud (i,j)
                pc=(i-1)*Nx+j-1;M[pl-1,pc-1]=1; # contribution de noeud (i,j-1)
                pc=(i-1)*Nx+j+1;M[pl-1,pc-1]=1; # contribution de noeud (i,j+1)
                pc=(i-2)*Nx+j;M[pl-1,pc-1]=1; # contribution de noeud (i-1,j)
                pc=(i)*Nx+j;M[pl-1,pc-1]=1; # contribution de noeud (i+1,j)
                b[pl-1]=-d**2*S[i-1,j-1]/k[i-1,j-1];
            # elif (i==1):
            #     # noeud sur le plafond y=0
            #     pc=pl;M[pl-1,pc-1]=1; # contribution de noeud (1,j)
            #     b[pl-1]=Tp;
            # elif (i==Ny):
            #     # noeud sur le plancher y=Ly
            #     pc=pl;M[pl-1,pc-1]=1; # contribution de noeud (Ny,j)
            #     b[pl-1]=Tp;  
            
            elif (i==1):
                # noeud sur le plafond y=0 avec condition de convection
                pc = pl; M[pl-1, pc-1] = 3 + 2*d*h/k[i-1,j-1]  # T0
                pc = pl + Nx; M[pl-1, pc-1] = -4                  # T1 (voisin en y)
                pc = pl + 2*Nx; M[pl-1, pc-1] = 1                   # T2 (voisin en y)
                b[pl-1] = 2*d*h*Ta/k[i-1,j-1]

            elif (i == Ny):
                # noeud sur le plancher y = Ly avec condition convective (Robin)
                # Ici, T0 correspond au nœud sur le plancher (i==Ny),
                # T1 est le nœud juste au-dessus (i==Ny-1),
                # T2 est le nœud encore au-dessus (i==Ny-2).
                pc = pl
                M[pl-1, pc-1] = 3.0 + 2.0*d*h/k[i-1, j-1]  # Coefficient de T0
                pc = pl - Nx  # déplacement d'une ligne vers le haut
                M[pl-1, pc-1] = -4.0                     # Coefficient de T1
                pc = pl - 2*Nx  # déplacement de deux lignes vers le haut
                M[pl-1, pc-1] = 1.0                      # Coefficient de T2
                b[pl-1] = 2.0*d*h*Ta/k[i-1, j-1]          # Second membre

            elif (j==1):
                # noeud à la surface externe du mur x=0
                pc=pl;M[pl-1,pc-1]=3+2*d*h/k[i-1,j-1]; # contribution de noeud (i,1)
                pc=(i-1)*Nx+j+1;M[pl-1,pc-1]=-4; # contribution de noeud (i,2)
                pc=(i-1)*Nx+j+2;M[pl-1,pc-1]=1; # contribution de noeud (i,3)
                b[pl-1]=2*d*h*Ta/k[i-1,j-1];
            elif (j==Nx):
                # noeud à la surface externe du mur x=Nx
                pc=pl;M[pl-1,pc-1]=3+2*d*h/k[i-1,j-1]; # contribution de noeud (i,Nx)
                pc=(i-1)*Nx+j-1;M[pl-1,pc-1]=-4; # contribution de noeud (i,Nx-1)
                pc=(i-1)*Nx+j-2;M[pl-1,pc-1]=1; # contribution de noeud (i,Nx-2)
                b[pl-1]=2*d*h*Ta/k[i-1,j-1];
            
            # Interface verticale mur-air côté intérieur à x = Lm
            # On suppose que le dernier nœud du mur est à j = int(Lm/d)
            # et le premier nœud intérieur est à j = int(Lm/d)+1.
            elif (j == int(Lm/d)+1):
                # Condition de continuité du flux : 
                # (k_m + k_a)*T(i,j) - k_m*T(i,j-1) - k_a*T(i,j+1) = 0
                pc = pl; 
                M[pl-1, pc-1] = km + ka
                pc = (i-1)*Nx + j-1; 
                M[pl-1, pc-1] = - km
                pc = (i-1)*Nx + j+1; 
                M[pl-1, pc-1] = - ka
                b[pl-1] = 0

            # Interface verticale mur-air côté intérieur à x = Lx - Lm
            # Supposons que le dernier nœud intérieur est à j = int((Lx-Lm)/d)
            # et que le premier nœud du mur est à j = int((Lx-Lm)/d)+1.
            elif (j == int((Lx-Lm)/d)+1):
                # On impose : (k_m + k_a)*T(i,j) - k_a*T(i,j-1) - k_m*T(i,j+1) = 0
                pc = pl; 
                M[pl-1, pc-1] = km + ka
                pc = (i-1)*Nx + j-1; 
                M[pl-1, pc-1] = - ka
                pc = (i-1)*Nx + j+1; 
                M[pl-1, pc-1] = - km
                b[pl-1] = 0

            # elif (j==1):
            #     # noeud à la surface externe du mur x=0
            #     pc=pl;M[pl-1,pc-1]=1 + (h * d / km); # contribution de noeud (i,1)
            #     pc=(i-1)*Nx+j+1;M[pl-1,pc-1]=-1; # contribution de noeud (i,2)
            #     pc=(i-1)*Nx+j+2;M[pl-1,pc-1]=1; # contribution de noeud (i,3)
            #     b[pl-1]=(h * d / km) * Ta;
            # elif (j==Nx):
            #     # noeud à la surface externe du mur x=Nx
            #     pc=pl;M[pl-1,pc-1]=1 + (h * d / km); # contribution de noeud (i,Nx)
            #     pc=(i-1)*Nx+j-1;M[pl-1,pc-1]=-1; # contribution de noeud (i,Nx-1)
            #     pc=(i-1)*Nx+j-2;M[pl-1,pc-1]=1; # contribution de noeud (i,Nx-2)
            #     b[pl-1]=(h * d / km) * Ta;
            # elif (j == int(Lm/d)):
            #     # Interface mur-air intérieur (x = Lm)
            #     pc = pl
            #     M[pl-1, pc-1] = km / ka + 1  # Contribution de T(Nm, j)
            #     M[pl-1, pl-2] = -km / ka  # Contribution de T(Nm-1, j)
            #     M[pl-1, pl+1] = -1  # Contribution de T(Nm+1, j)

            # elif (j == int((Lx-Lm)/d)):
            #     # Interface mur-air intérieur (x = Lx - Lm)
            #     pc = pl
            #     M[pl-1, pc-1] = km / ka + 1
            #     M[pl-1, pl-2] = -1
            #     M[pl-1, pl+1] = -km / ka
            # else:
            #     print('Erreur dans la définition de la matrice de coefficients');
    

    toc=time.time_ns();
    tini_ar[ci]=(toc-tic)/1.0e9; #temps en [s]  
    
    tic=time.time_ns();
    M_csr = M.tocsr()
    T_sparse = spsolve(M_csr, b.ravel())   # b.ravel() pour obtenir un vecteur 1D
    T = T_sparse.reshape((Nx*Ny, 1))       # on remet en forme (Nx*Ny,1)
    toc=time.time_ns();
    tinv_ar[ci]=(toc-tic)/1.0e9; #temps en [s]  
    
    mem_ar[ci]=8*(Nx*Ny)**2;
    Tr=np.reshape(T,(Ny,Nx),order='C');
    
    Tm_ar[ci]=Tr[int(np.rint(Ly/d/2+1))-1,int(np.rint(Lx/d/2+1))-1]; # température au milieu du domaine de calcul

plt.figure(1)
plt.pcolor(np.arange(0,Nx,1)*d,np.arange(0,Ny,1)*d,S);
plt.colorbar(mappable=None, cax=None, ax=None);
plt.title('S(x,y) [W/$m^3$]')
plt.xlabel('x [m]')    
plt.ylabel('y [m]')


plt.figure(2)
plt.pcolor(np.arange(0,Nx,1)*d,np.arange(0,Ny,1)*d,k);
plt.colorbar(mappable=None, cax=None, ax=None);
plt.title('k(x,y) [W/($m^2\cdot$K)]')
plt.xlabel('x [m]')    
plt.ylabel('y [m]')


plt.figure(3)
plt.pcolor(np.arange(0,Nx,1)*d,np.arange(0,Ny,1)*d,Tr);
plt.colorbar(mappable=None, cax=None, ax=None);
plt.title('T(x,y) [$^o$C]')
plt.xlabel('x [m]')    
plt.ylabel('y [m]')

plt.figure(4)
plt.loglog(d_ar[::-1],mem_ar[::-1]/1024.0**3,'-o')
plt.title('Exigences de mémoire')
plt.xlabel('Pas $d_x=d_y$ [m]')
plt.ylabel('Mémoire [Gb]')

plt.figure(5)
Err_ar=abs(Tm_ar[:-1:]-Tm_ar[1::]);
d_Err_ar=d_ar[1::]; # Definiton d'erreur Err(delta)=|Tm(2*delta)-Tm(delta)|
plt.loglog(d_Err_ar[::-1],Err_ar[::-1],'-o')
plt.title('Erreur de calcul')
plt.xlabel('Pas $d_x=d_y$ [m]')
plt.ylabel('Err [$^o$C]')

plt.figure(6)
plt.loglog(d_ar[::-1],tini_ar[::-1],'-bo',d_ar[::-1],tinv_ar[::-1],'-r*')
plt.title('Temps de calcul(initialisation et inversion)')
plt.xlabel('Pas $d_x=d_y$ [m]')
plt.ylabel('t [s]')
plt.legend(['$t_{initialisation}$','$t_{inversion}$'])

print("\n=== Résultats numériques ===")
print("Pas (cm)   T_m (°C)    Mémoire (octets)")
for i in range(fact_ar.size):
    d_cm = d_ar[i]*100
    Tm   = Tm_ar[i]
    mem_bytes = mem_ar[i]
    print(f"{d_cm:8.3f}   {Tm:8.3f}   {mem_bytes:12.0f}")


print("\n=== Erreur entre deux pas successifs ===")
print("delta1 (cm)   delta2 (cm)   Err (°C)")
for i in range(fact_ar.size - 1):
    d1_cm = d_ar[i]*100
    d2_cm = d_ar[i+1]*100
    print(f"{d1_cm:8.3f}    {d2_cm:8.3f}    {Err_ar[i]:8.4f}")

plt.figure()
plt.pcolor(
    np.arange(0, Nx, 1)*d_ar[-1],  # coordonnée X
    np.arange(0, Ny, 1)*d_ar[-1],  # coordonnée Y
    Tr
)
plt.colorbar()
plt.title(f"Distribution de T pour d={d_ar[-1]*100:.3f} cm")
plt.xlabel("x [m]")
plt.ylabel("y [m]")

print("\n=== Résultats du maillage ===")
for i in range(fact_ar.size):
    print(f"delta = {d_ar[i]:.4f} m => Nx={int(np.rint(Lx/d_ar[i]+1))}, Ny={int(np.rint(Ly/d_ar[i]+1))}, "
          f"mem={mem_ar[i]:.2e} octets, t_ini={tini_ar[i]:.3g}s, t_inv={tinv_ar[i]:.3g}s, Tm={Tm_ar[i]:.3g}°C")

# Tableau N_ar = Nx * Ny
N_ar = np.zeros(fact_ar.size, dtype=np.double)
for i in range(fact_ar.size):
    Nx_ = int(np.rint(Lx/d_ar[i]+1))
    Ny_ = int(np.rint(Ly/d_ar[i]+1))
    N_ar[i] = Nx_*Ny_

# Ajustement mémoire en log-log
import numpy as np
logN = np.log(N_ar)
logMem = np.log(mem_ar)

coeff_mem = np.polyfit(logN, logMem, 1)
p_mem = coeff_mem[0]
A_mem = np.exp(coeff_mem[1])

print(f"\n=== Ajustement mémoire : mem ~ A * N^p ===")
print(f"p_mem = {p_mem:.3f}, A_mem = {A_mem:.3e}")

# Ajustement t_ini en log-log
logTini = np.log(tini_ar)
coeff_tini = np.polyfit(logN, logTini, 1)
p_tini = coeff_tini[0]
A_tini = np.exp(coeff_tini[1])

print(f"\n=== Ajustement t_ini : t_ini ~ A * N^p ===")
print(f"p_tini = {p_tini:.3f}, A_tini = {A_tini:.3e}")

# Ajustement t_inv en log-log
logTinv = np.log(tinv_ar)
coeff_tinv = np.polyfit(logN, logTinv, 1)
p_tinv = coeff_tinv[0]
A_tinv = np.exp(coeff_tinv[1])

print(f"\n=== Ajustement t_inv : t_inv ~ A * N^p ===")
print(f"p_tinv = {p_tinv:.3f}, A_tinv = {A_tinv:.3e}")

# Ajustement erreur (si on veut en fonction de delta)
# 1 point de moins dans Err_ar, donc on prend d_ar[1:]
if fact_ar.size > 1:
    logD = np.log(d_ar[1:])
    logErr = np.log(Err_ar)
    coeff_err = np.polyfit(logD, logErr, 1)
    p_err = coeff_err[0]
    A_err = np.exp(coeff_err[1])
    print(f"\n=== Ajustement Erreur : Err ~ A * d^p ===")
    print(f"p_err = {p_err:.3f}, A_err = {A_err:.3e}")

plt.show()

