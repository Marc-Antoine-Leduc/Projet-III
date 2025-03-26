import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

def makeBasicAnimation(mod_psis, Nt, L):
    """
    Créer une animation avec le domaine seulement.

    Args :
        mod_psis (array) : Vecteur de fonctions d'onde discrétisées.
        Nt (int) : Nombre de pas de temps.
        L (int) : Grandeur du domaine de simulation.

    Returns :
        anim (plot) : Animation de la fonction d'onde.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0,L), ylim=(0,L))
    img = ax.imshow(mod_psis[0]**2, extent=[0,L,0,L], cmap=plt.get_cmap("hot"), vmin=0, zorder=1)

    def animate(i):
        img.set_data(mod_psis[i]**2)
        img.set_clim(vmin=0, vmax=np.max(mod_psis[i]**2))  # Échelle dynamique par frame
        img.set_zorder(1)
        return img,

    anim = FuncAnimation(fig, animate, interval=1, frames=np.arange(0, Nt, 2), repeat=False, blit=False)

    output_dir = r"."
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "basicAnimation.mp4")
    print(f"Enregistrement de l'animation dans : {output_file}")
    anim.save(output_file, writer="ffmpeg", fps=60)
    plt.show()

    return anim

def makeAnimationForSlits(mod_psis, v, L, Nt):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    fig, ax = plt.subplots()
    
    img_wave = ax.imshow(mod_psis[0]**2, extent=[0,L,0,L], origin='lower',
                         cmap='hot', vmin=0, vmax=np.max(mod_psis[0]**2))
    
    img_pot = ax.imshow(v.real, extent=[0,L,0,L], origin='lower',
                        cmap='gray', alpha=0.3,  
                        vmin=0, vmax=np.max(v.real))
    
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    
    def update(frame):
        # Met à jour la fonction d’onde
        wave_sq = mod_psis[frame]**2
        img_wave.set_data(wave_sq)
        img_wave.set_clim(vmin=0, vmax=np.max(wave_sq))
        return (img_wave, img_pot)
    
    anim = FuncAnimation(fig, update, frames=Nt, interval=50, blit=False)

    output_dir = r"."
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "basicAnimation.mp4")
    print(f"Enregistrement de l'animation dans : {output_file}")
    anim.save(output_file, writer="ffmpeg", fps=60)
    plt.show()
    
    return anim
####################################################

# def makeAnimationForSlits(mod_psis, j0, i0, i1, i2, i3, Dy, Nt, w, L):
#     """
#     Créer une animation pour les doubles fentes. 

#     Args :
#         mod_psis (array) : Vecteur de fonctions d'onde discrétisées.
#         jo (float) : Extrémité gauche de la simulation.
#         j1 (float) : Extrémité droite de la simulation.
#         i0 (float) : Extrémité basse de la fente la plus basse.
#         i1 (float) : Extrémité haute de la fente la plus basse.
#         i2 (float) : Extrémité basse de la fente la plus haute.
#         i3 (float) : Extrémité haute de la fente la plus haute.
#         Dy (float) : Pas d'espace en y.
#         Nt (int) : Nombre de pas de temps.
#         w (float) : Épaisseur du mur.
#         L (int) : Grandeur du domaine de simulation.

#     Returns :
#         anim (plot) : Animation de la fonction d'onde.
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, xlim=(0,L), ylim=(0,L))
   
#     img = ax.imshow(mod_psis[0]**2, extent=[0,L,0,L], cmap=plt.get_cmap("hot"), vmin=0, zorder=1)

#     # Ajout des doubles fentes (murs)
#     slitcolor = "w"
#     slitalpha = 0.08 # Transparance des murs.
#     wall_bottom = Rectangle((j0*Dy,0),     w, i3*Dy,      color=slitcolor, zorder=50, alpha=slitalpha)
#     wall_middle = Rectangle((j0*Dy,i2*Dy), w, (i1-i2)*Dy, color=slitcolor, zorder=50, alpha=slitalpha)
#     wall_top    = Rectangle((j0*Dy,i0*Dy), w, i3*Dy,      color=slitcolor, zorder=50, alpha=slitalpha)

#     ax.add_patch(wall_bottom)
#     ax.add_patch(wall_middle)
#     ax.add_patch(wall_top)

#     def animate(i):
#         img.set_data(mod_psis[i]**2)
#         img.set_clim(vmin=0, vmax=np.max(mod_psis[i]**2))  # Échelle dynamique par frame
#         img.set_zorder(1)
#         return img,

#     anim = FuncAnimation(fig, animate, interval=1, frames=np.arange(0, Nt, 2), repeat=False, blit=False)

#     output_dir = r"."
#     os.makedirs(output_dir, exist_ok=True)
#     output_file = os.path.join(output_dir, "doubleSlitsAnimation.mp4")
#     print(f"Enregistrement de l'animation dans : {output_file}")
#     anim.save(output_file, writer="ffmpeg", fps=60)
#     plt.show()

#     return anim
####################################################

def makeAnimationForCristal(mod_psis, j0, i0, i1, i2, i3, Dy, Nt, w, L):
    """
    Créer une animation pour le cristal 2D.

    Args :
        mod_psis (array) : Vecteur de fonctions d'onde discrétisées.
        jo (float) : Extrémité gauche du mur.
        j1 (float) : Extrémité droite du mur.
        i0 (float) : Extrémité basse de la fente la plus basse.
        i1 (float) : Extrémité haute de la fente la plus basse.
        i2 (float) : Extrémité basse de la fente la plus haute.
        i3 (float) : Extrémité haute de la fente la plus haute.
        Dy (float) : Pas d'espace en y.
        Nt (int) : Nombre de pas de temps.
        w (float) : Épaisseur du mur.
        L (int) : Grandeur du domaine de simulation.

    Returns :
        anim (plot) : Animation de la fonction d'onde.
    """
    fig = plt.figure() 
    ax = fig.add_subplot(111, xlim=(0,L), ylim=(0,L)) 

    img = ax.imshow(mod_psis[0], extent=[0,L,0,L], cmap=plt.get_cmap("hot"), vmin=0, vmax=np.max(mod_psis), zorder=1) 

    # On crée les murs.
    slitcolor = "w" 
    slitalpha = 0.08 # Transparence des murs.
    wall_bottom = Rectangle((j0*Dy,0),     w, i3*Dy,      color=slitcolor, zorder=50, alpha=slitalpha) # (x0, y0), width, height
    wall_middle = Rectangle((j0*Dy,i2*Dy), w, (i1-i2)*Dy, color=slitcolor, zorder=50, alpha=slitalpha)
    wall_top    = Rectangle((j0*Dy,i0*Dy), w, i3*Dy,      color=slitcolor, zorder=50, alpha=slitalpha)

    ax.add_patch(wall_bottom)
    ax.add_patch(wall_middle)
    ax.add_patch(wall_top)

    def animate(i):
        img.set_data(mod_psis[i]**2)  
        img.set_clim(vmin=0, vmax=np.max(mod_psis[i]**2))  # Échelle dynamique
        img.set_zorder(1)
        return img, 


    anim = FuncAnimation(fig, animate, interval=1, frames =np.arange(0,Nt,2), repeat=False, blit=0) # We generate the animation.# Generamos la animación.

    plt.show()

    anim.save('./animationsName.mp4', writer="ffmpeg", fps=60)


    return anim
####################################################

def diffractionPatron(mod_psis, L, Ny):
    """
    Crée une figure montrant le patron de diffraction de la densité de probabilité cumulative.

    Args:
        mod_psis (list of arrays): Liste des modules de la fonction d'onde discrétisée à chaque pas de temps.
        L (float): Longueur du domaine.
        Ny (int): Nombre de points dans la direction y du domaine.

    Returns:
        cumulative_intensity (array): Intensité cumulative (somme sur le temps) sur l'écran (à x = L).
    """
    import numpy as np
    import matplotlib.pyplot as plt


    cumulative_intensity = np.zeros(mod_psis[0].shape[0])
    
    # Somme cumulative de l'intensité sur l'écran à chaque instant
    for psi in mod_psis:
        cumulative_intensity += np.abs(psi[:, -1])**2

    y_screen = np.linspace(0, L, mod_psis[0].shape[0])
    
    # patron = int((2.8*len(mod_psis))/3)
    final_psi = mod_psis[-1]  # Dernière étape temporelle
    screen_intensity = np.abs(final_psi[:, -1])**2  # Intensité (|psi|^2) sur le bord droit
    y_screen = np.linspace(0, L, Ny-2)  # Coordonnées y le long de l’écran

    # Affichage du patron de diffraction
    plt.figure(figsize=(8, 6))
    plt.plot(y_screen, cumulative_intensity, label='Patron de diffraction cumulé')
    plt.plot(y_screen, screen_intensity, label="Patron de diffraction simulé")
    plt.plot(y_screen, theoreticalIntensity(np.linspace(-L/2, L/2, Ny-2), s=0.8, a=0.2, L=L, k=5*np.pi), label="Patron de diffraction théorique", linestyle="dashed")
    plt.xlabel('Position y')
    plt.ylabel('Intensité cumulative (|ψ|²)')
    plt.title("Patron de diffraction cumulé sur l'écran à x = L")
    plt.grid(True)
    plt.legend()
    plt.show()

    return cumulative_intensity


####################################################

def theoreticalIntensity(y, s, a, L, k):
    """
    Fonction théorique pour le patron de diffraction des fentes de Young.

    Args :
        y (array) : Plage de valeurs.
        s (float) : Distance entre les fentes.
        a (float) : Largeur des fentes.
        L (int) : Longueur du domaine.
        k (float) : Vecteur d'onde.
    
    Returns :
        (cos_term**2) * (sinc_term**2) (array) : Fonction théorique.
    """
    lambda_ = 2*np.pi / k
    sinc_term = np.sinc((np.pi * a * y) / (lambda_ * L))
    cos_term = np.cos((np.pi * s * y) / (lambda_ * L))
    return (cos_term**2) * (sinc_term**2) 
####################################################

def theoreticalIntensity(y, s, a, L, k):
    """
    Fonction théorique pour le patron de diffraction des fentes de Young.

    Args :
        y (array) : Plage de valeurs.
        s (float) : Distance entre les fentes.
        a (float) : Largeur des fentes.
        L (int) : Longueur du domaine.
        k (float) : Vecteur d'onde.
    
    Returns :
        (cos_term**2) * (sinc_term**2) (array) : Fonction théorique.
    """
    lambda_ = 2*np.pi / k
    sinc_term = np.sinc((np.pi * a * y) / (lambda_ * L))
    cos_term = np.cos((np.pi * s * y) / (lambda_ * L))
    return (cos_term**2) * (sinc_term**2) 
####################################################

# def saveData(mod_psis):
#     """
#     Enregistrer l'animation.

#     Args : 
#         mod_psis (plot) : Animation
#     """
    
#     # We transform the 3D array into a 2D array to save it with numpy.savetxt.
#     mod_psis_reshaped = np.asarray(mod_psis).reshape(np.asarray(mod_psis).shape[0], -1) 
    
#     # We save the 2D array as a text file.
#     np.savetxt(r"C:\Users\leduc\OneDrive\Documents\École\Université\Session 6\PHS3903 - Projet III\Résultats", mod_psis_reshaped)
    
# ####################################################

# def obtainData(Ny): 
#     """
#     Obtenir le data d'un vecteur de fonctions d'onde déjà créé.

#     Args :
#         Ny (int) : Grandeur du grillage en y.
#     """
    
#     # To obtain the data from the text file already created earlier.
#     loaded_mod_psis = np.loadtxt("mod_psis_data.txt")
    
#     # The loaded_mod_psis array is a 2D array, we need to return it to its original form.

#     mod_psisshape2 = Ny-2

#     # We finally obtain our mod_psis array.

#     mod_psis = loaded_mod_psis.reshape( 
#         loaded_mod_psis.shape[0], loaded_mod_psis.shape[1] // mod_psisshape2, mod_psisshape2) 
    
#     ## For deleting the auxiliary 2D array.
#     # del loaded_mod_psis
    
