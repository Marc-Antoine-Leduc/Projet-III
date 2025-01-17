import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt

def MonteCarlo(Dim,pt_par_essai):
    R=1 #valeur rayon
    Nb_essai=100 # Nombre d'essais par simulation
    a = 2 # Dimension de la boîte cubique dans laquelle les points aléatoires seront générés

    #qql variables pour les itérations
    Dimlen = len(Dim)
    pt_par_essailen = len(pt_par_essai)
    V = np.zeros((Dimlen, pt_par_essailen))
    Err=np.zeros((Dimlen, pt_par_essailen))
    Errm=np.zeros((Dimlen, pt_par_essailen))
    Inc=np.zeros((Dimlen, pt_par_essailen))


    for d in range(0, Dimlen):

        D = Dim[d]  # Dimension
        Vtot = a**D # Volume du domaine
        Vth = np.pi**(D/2) / gamma((D/2) + 1) # Volume théorique
        print(Vth)
        

        for n in range(0, pt_par_essailen):
            Ntot = pt_par_essai[n]  # Nombre de points

            Vind = np.zeros(Nb_essai) # Volumes calculés pour chaque essai individuel
            Err_rel = np.zeros(Nb_essai)

            for k in range(0,Nb_essai): # Boucle sur les essais
                # Génération des nombres aléatoires (distribution uniforme)
                np.random.seed() # Initialise le générateur de nombres pseudo-aléatoires afin de ne pas toujours produire la même séquence à l'ouverture de Python...
                pts = np.random.uniform(low=-1, high=1, size=(Ntot, D)) # Coordonnées des points
                

                # Calcul du volume
                
                distances= np.sqrt(np.sum(pts**2, axis=1))
                
                Nint = np.sum(distances <= R)  # Nombre de points à l'intérieur
                Vcal= Nint / Ntot * Vtot # Volume calculé pour cet essai
                Vind[k] = Vcal
                erreur_relative = np.abs(Vcal - Vth) / Vth 
                Err_rel[k]=erreur_relative*100
            mvind=np.mean(Vind)
            V[d, n] = mvind # Volume moyenné sur l'ensemble des essais
            Incertiture_rel=np.std(Vind)/mvind
            Inc[d,n]=Incertiture_rel*100
            Errm[d,n]=np.mean(Err_rel)
            

    return V, Inc,Errm


N_values = [3, 6]
Ntot_list = [100, 200, 400, 800, 1600]

V,Inc,Err= MonteCarlo(N_values,Ntot_list)

print(V)
print(Err)
print(Inc)





plt.plot(Ntot_list,Err[0],label='N=3', marker='o')
plt.plot(Ntot_list,Err[1],label='N=6',marker='o')

plt.title("L'erreur relative E (%) sur le volume calculé en fonction de N total en pour chaque N-sphère")

plt.xscale('log')
plt.yscale('log')
plt.xlim(100,1600)

plt.xlabel("Le nombre de points par essai")
plt.ylabel("L'erreur relative (%)")
x = np.log(Ntot_list)
y = np.log(Err[0])
coefficients = np.polyfit(x, y, 1)
polynomial = np.poly1d(coefficients)
plt.plot(Ntot_list, np.exp(polynomial(x)), '--', label='Logarithmic Regression N=3')
equation = f'y = {np.exp(coefficients[1]):.2f}  {coefficients[0]:.2f}x'
plt.text(210, np.exp(polynomial(np.log(200))) * 1.1, equation, fontsize=12)
s = np.log(Ntot_list)
t = np.log(Err[1])
coefficients2 = np.polyfit(s, t, 1)

polynomial = np.poly1d(coefficients2)
plt.plot(Ntot_list, np.exp(polynomial(x)), '--', label='Logarithmic Regression N=6')

equation2 = f'y = {np.exp(coefficients2[1]):.2f}  {coefficients2[0]:.2f}x'
plt.text(210, np.exp(polynomial(np.log(200))) * 1.1, equation2, fontsize=12)
print(coefficients)
print(coefficients2)


plt.legend()


plt.show()


