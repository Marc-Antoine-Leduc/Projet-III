import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import sparse
from scipy.sparse.linalg import spsolve
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

class WaveFunction(object):

    def __init__(self, x, y, psi_0, V, dt, hbar=1, m=1, t0=0.0):
        self.x = np.array(x)
        self.y = np.array(y)
        self.psi = np.array(psi_0, dtype=np.complex128)
        self.V = np.array(V, dtype=np.complex128)
        self.dt = dt
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.hbar = hbar
        self.m = m
        self.t = t0
        self.nb_frame = 300
        self.nbr_level = 200

        alpha = dt/(4*self.dx**2)
        self.alpha = alpha
        self.size_x = len(x)
        self.size_y = len(y)
        dimension = self.size_x*self.size_y

        #Building the first matrix to solve the system (A from Ax_{n+1}=Mx_{n})
        N = (self.size_x-1)*(self.size_y-1)
        size = 5*N + 2*self.size_x + 2*(self.size_y-2)
        I = np.zeros(size)
        J = np.zeros(size)
        K = np.zeros(size, dtype=np.complex128)

        k = 0
        for i in range(0,self.size_y):
            for j in range(0,self.size_x):
                #Condition aux frontières nulles aux extrémités (en y)
                if i==0 or i==(self.size_y-1):
                    I[k] = i + j*self.size_y
                    J[k] = i + j*self.size_y
                    K[k] = 1
                    k += 1

                #Conditions aux frontières nulles aux extrémités (en x)
                elif j==0 or j==(self.size_x-1):
                    I[k] = i + j*self.size_y
                    J[k] = i + j*self.size_y
                    K[k] = 1
                    k += 1

                #Points à l'intérieur du domaine
                else:
                    #Point central (i,j)
                    I[k] = i + j*self.size_y
                    J[k] = i + j*self.size_y
                    K[k] = 1.0j - 4*alpha - V[i+j*self.size_y]*dt/2
                    k += 1

                    #Point (i-1,j)
                    I[k] = i + j*self.size_y
                    J[k] = (i-1) + j*self.size_y
                    K[k] = alpha
                    k += 1

                    #Point (i+1,j)
                    I[k] = i + j*self.size_y
                    J[k] = (i+1) + j*self.size_y
                    K[k] = alpha
                    k += 1

                    #Point (i,j-1)
                    I[k] = i + j*self.size_y
                    J[k] = i + (j-1)*self.size_y
                    K[k] = alpha
                    k += 1

                    #Point (i,j+1)
                    I[k] = i + j*self.size_y
                    J[k] = i + (j+1)*self.size_y
                    K[k] = alpha
                    k += 1

        self.Mat1 = sparse.coo_matrix((K,(I,J)),shape=(dimension,dimension)).tocsc()

        #Building the second matrix to solve the system (M from Ax_{n+1}=Mx_{n})
        I = np.zeros(size)
        J = np.zeros(size)
        K = np.zeros(size, dtype=np.complex128)

        k = 0
        for i in range(0,self.size_y):
            for j in range(0,self.size_x):
                #Condition aux frontières nulles aux extrémités (en y)
                if i==0 or i==(self.size_y-1):
                    I[k] = i + j*self.size_y
                    J[k] = i + j*self.size_y
                    K[k] = 0
                    k += 1

                #Conditions aux frontières nulles aux extrémités (en x)
                elif j==0 or j==(self.size_x-1):
                    I[k] = i + j*self.size_y
                    J[k] = i + j*self.size_y
                    K[k] = 0
                    k += 1

                #Points à l'intérieur du domaine
                else:
                    #Point central (i,j)
                    I[k] = i + j*self.size_y
                    J[k] = i + j*self.size_y
                    K[k] = 1.0j + 4*alpha + V[i+j*self.size_y]*dt/2
                    k += 1

                    #Point (i-1,j)
                    I[k] = i + j*self.size_y
                    J[k] = (i-1) + j*self.size_y
                    K[k] = -alpha
                    k += 1

                    #Point (i+1,j)
                    I[k] = i + j*self.size_y
                    J[k] = (i+1) + j*self.size_y
                    K[k] = -alpha
                    k += 1

                    #Point (i,j-1)
                    I[k] = i + j*self.size_y
                    J[k] = i + (j-1)*self.size_y
                    K[k] = -alpha
                    k += 1

                    #Point (i,j+1)
                    I[k] = i + j*self.size_y
                    J[k] = i + (j+1)*self.size_y
                    K[k] = -alpha
                    k += 1

        self.Mat2 = sparse.coo_matrix((K,(I,J)),shape=(dimension,dimension)).tocsc()


    def getProbability(self):
        return (abs(self.psi))**2

    def computeNorm(self):
        return np.trapz(np.trapz((self.getProbability()).reshape(self.size_y,self.size_x), self.x).real, self.y).real

    def step(self):

        mod_psis = [np.abs(spsolve(self.Mat1, self.Mat2.dot(self.psi)))] 
        
        for i in range(1,self.nb_frame):
            #Update the state
            self.psi = spsolve(self.Mat1, self.Mat2.dot(self.psi))

            mod_psis.append(self.psi.reshape(self.size_x-2, self.size_y-2))

            #Update time
            self.t += self.dt

        return mod_psis


        


#%%
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle

plt.rcParams.update({'font.size': 7})

# i) Gaussian wave packet
def gauss(x, y, delta_x, delta_y, x0, y0, kx0, ky0):
    return 1/(2*delta_x**2*np.pi)**(1/4) * 1/(2*delta_y**2*np.pi)**(1/4) * np.exp(-((x-x0)/(2*delta_x)) ** 2) * np.exp(-((y-y0)/(2*delta_y)) ** 2) * np.exp( 1.j * (kx0*x + ky0*y))

# ii) Heaviside function for the square potential
def potentielHeaviside(V0, x0, xf, y0, yf, x, y):
    V = np.zeros(len(x)*len(y))
    size_y = len(y)
    for i,yi in enumerate(y):
        for j,xj in enumerate(x):
            if (xj >= x0) and (xj <= xf) and (yi >= y0) and (yi <= yf):
                V[i+j*size_y] = V0
            else:
                V[i+j*size_y] = 0
    return V

# iii)
def intervalle(max_list,min_list,list_ref,n=3):
    return [round(i, -int(np.floor(np.log10(i))) + (n - 1))  for i in list_ref if (i < max_list) and (i > min_list) ]

# iv) Analytical model
def analyticModulus(x, y, a, x0, y0, kx0, ky0, t):
    sigma = np.sqrt(a**2 + t**2/(4*a**2))
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/2*((x-x0-(kx0)*t)/sigma)**2) * 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/2*((y-y0-(ky0)*t)/sigma)**2)

def computeError(z1,z2,x,y):
    return np.trapz(np.trapz(abs(z1-z2), x).real, y).real

#%%

# specify time steps and duration
dt = 0.005

# specify constants
hbar = 1.0   # planck's constant
m = 1.0      # particle mass

# specify range in x coordinate
x_min = -8
x_max = 13
dx = 0.08
x = np.arange(x_min, x_max+dx, dx)

# specify range in y coordinate
y_min = -12
y_max = 12
dy = dx
y = np.arange(y_min, y_max+dy, dy)

ni = 250
xi = np.linspace(x.min(),x.max(),ni)
yi = np.linspace(y.min(),y.max(),ni)
xig, yig = np.meshgrid(xi, yi)

#Create the potential
V0 = 400

x01 = 0
xf1 = 0.3
y01 = y.min()
yf1 = -2.85

#x0m = x01
#xfm = xf1
#y0m = -0.5
#yfm = 0.5

x02 = x01
xf2 = xf1
y02 = -yf1
yf2 = y.max()

V_xy = potentielHeaviside(V0,x01,xf1,y01,yf1,x,y) + potentielHeaviside(V0,x02,xf2,y02,yf2,x,y) #+ potentielHeaviside(V0,x0m,xfm,y0m,yfm,x,y)

#V_xy = np.zeros(len(x)*len(y))

#Specify the parameter of the initial gaussian packet
x0 = -5
y0 = 0
#kx0 = 2*np.sqrt(11)
kx0 = 20
ky0 = 0
delta_x = 0.7
delta_y = 0.7

#Create the initial wave packet
size_x = len(x)
size_y = len(y)
xx, yy = np.meshgrid(x,y)
psi_0 = gauss(xx, yy, delta_x, delta_y, x0, y0, kx0, ky0).transpose().reshape(size_x*size_y)

# Define the Schrodinger object which performs the calculations
S = WaveFunction(x=x, y=y, psi_0=psi_0, V=V_xy, dt=dt, hbar=hbar,m=m)
S.psi = S.psi/S.computeNorm()

#%%
fig = plt.figure(figsize=(11,8))
plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])
plt.xlabel(r"x ($a_0$)", fontsize = 16)
plt.ylabel(r"y ($a_0$)", fontsize = 16)

#Initial plotting
t = 0
z = S.getProbability().reshape(size_x,size_y).transpose()

#Draw the potential
plt.text(0.02, 0.92, r"t = 0.0000  (u.a.)".format(S.t), color='white', fontsize=12)
plt.vlines(x01, y01, yf1, colors='white', zorder=2)
plt.vlines(xf1, y01, yf1, colors='white', zorder=2)
plt.vlines(x02, y02, yf2, colors='white', zorder=2)
plt.vlines(xf2, y02, yf2, colors='white', zorder=2)
plt.hlines(yf1, x01, xf1, colors='white', zorder=2)
plt.hlines(y02, x01, xf1, colors='white', zorder=2)

x_desired = 11
k = abs(x-x_desired).argmin()

t_vec = np.arange(0,S.nb_frame*dt,dt)
coupe = np.zeros((S.nb_frame,len(z[:,k])))

#Create animation
z = S.getProbability().reshape(size_x,size_y).transpose()

#plotting
#i) first plot
level = np.linspace(0,z.max(),S.nbr_level)
cset = plt.contourf(xx, yy, z, levels=level, cmap=plt.cm.jet,zorder=1)
plt.xlabel(r"x ($a_0$)", fontsize = 16)
plt.ylabel(r"y ($a_0$)", fontsize = 16)

#Draw the potential
plt.text(0.02, 0.92, r"t = {0:.3f} (u.a.)".format(S.t), color='white', fontsize=12)
plt.vlines(x01, y01, yf1, colors='white', zorder=2)
plt.vlines(xf1, y01, yf1, colors='white', zorder=2)
plt.vlines(x02, y02, yf2, colors='white', zorder=2)
plt.vlines(xf2, y02, yf2, colors='white', zorder=2)
plt.hlines(yf1, x01, xf1, colors='white', zorder=2)
plt.hlines(y02, x01, xf1, colors='white', zorder=2)

#Adjust the colorbar
cbar1 = fig.colorbar(cset)
ticks = intervalle(z.max(), 0, np.linspace(0,4*z.max(),50))
cbar1.set_ticks(ticks)
cbar1.set_ticklabels(ticks)



interval = 0.001
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def makeAnimation(mod_psis, L, Nt):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import os

    fig, ax = plt.subplots()
    
    img_wave = ax.imshow(mod_psis[0]**2, extent=[0, L, 0, L], origin='lower',
                         cmap='hot', vmin=0, vmax=np.max(mod_psis[0]**2))

    
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.legend()

    def update(frame):
        wave_sq = mod_psis[frame]**2
        img_wave.set_data(wave_sq)
        img_wave.set_clim(vmin=0, vmax=np.max(wave_sq))
        return (img_wave)
    
    anim = FuncAnimation(fig, update, frames=Nt, interval=50, blit=False)
    
    plt.show()
    
    output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "basicAnimation.mp4")
    print(f"Enregistrement de l'animation dans : {output_file}")
    anim.save(output_file, writer="ffmpeg", fps=60)

    return anim

anim = makeAnimation(mod_psis=S.step(), L=S.size_x, Nt=S.nb_frame)

anim.save('2D_2slit_dx={0}_dt={1}_yf1={2}_k={3}.mp4'.format(dx,dt,abs(yf1),kx0), fps=15, extra_args=['-vcodec', 'libx264'])

with open("2_slit_dx={0}_dt={1}_yf1={2}_k={3}.pkl".format(dx,dt,abs(yf1),kx0), 'wb') as pickleFile:
    pickle.dump(coupe, pickleFile)
    pickleFile.close()

exit()
plt.show()
