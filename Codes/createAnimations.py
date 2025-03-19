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
    fig = plt.figure() # On crée la figure.
    ax = fig.add_subplot(111, xlim=(0,L), ylim=(0,L)) # On ajoute un subplot avec le domaine de longueur L.

    img = ax.imshow(mod_psis[0], extent=[0,L,0,L], cmap=plt.get_cmap("hot"), vmin=0, vmax=np.max(mod_psis), zorder=1) # Module de la fonction d'onde 2D.

    def animate(i):
        """
        On crée l'animation.
        """
        img.set_data(mod_psis[i]**2) # Mettre les modules de la fonction d'onde dans img.
        img.set_zorder(1)
        
        return img,


    anim = FuncAnimation(fig, animate, interval=1, frames=np.arange(0, Nt, 2), repeat=False, blit=False)

    # Sauvegarde avant d'afficher l'animation
    output_dir = r"C:\Users\leduc\OneDrive\Documents\École\Université\Session 6\PHS3903 - Projet III\Résultats"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "test2.mp4")

    print(f"Enregistrement de l'animation dans : {output_file}")
    anim.save(output_file, writer="ffmpeg", fps=60)

    plt.show()

    ## Enregistrer l'animation (Ubuntu).
    # anim.save('./animationsName.mp4', writer="ffmpeg", fps=60)

    return anim
####################################################

def makeAnimationForSlits(mod_psis, j0, i0, i1, i2, i3, Dy, Nt, w, L):
    """
    Créer une animation pour les doubles fentes. 

    Args :
        mod_psis (array) : Vecteur de fonctions d'onde discrétisées.
        jo (float) : Extrémité gauche de la simulation.
        j1 (float) : Extrémité droite de la simulation.
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
    fig = plt.figure() # We create the figure.
    ax = fig.add_subplot(111, xlim=(0,L), ylim=(0,L)) # We add the subplot to the figure.

    img = ax.imshow(mod_psis[0], extent=[0,L,0,L], cmap=plt.get_cmap("hot"), vmin=0, vmax=np.max(mod_psis), zorder=1) # Here the modulus of the 2D wave function shall be represented.

    # We paint the walls of the double slit with rectangles.
    slitcolor = "w" # Color of the rectangles.
    slitalpha = 0.08 # Transparency of the rectangles.
    wall_bottom = Rectangle((j0*Dy,0),     w, i3*Dy,      color=slitcolor, zorder=50, alpha=slitalpha) # (x0, y0), width, height
    wall_middle = Rectangle((j0*Dy,i2*Dy), w, (i1-i2)*Dy, color=slitcolor, zorder=50, alpha=slitalpha)
    wall_top    = Rectangle((j0*Dy,i0*Dy), w, i3*Dy,      color=slitcolor, zorder=50, alpha=slitalpha)

    # We add the rectangular patches to the plot.
    ax.add_patch(wall_bottom)
    ax.add_patch(wall_middle)
    ax.add_patch(wall_top)

    # We define the animation function for FuncAnimation.

    def animate(i):
        
        img.set_data(mod_psis[i]**2) # Fill img with the modulus data of the wave function.
        img.set_zorder(1)
        
        return img, # We return the result ready to use with blit=True.


    anim = FuncAnimation(fig, animate, interval=1, frames=np.arange(0, Nt, 2), repeat=False, blit=False)

    # Sauvegarde avant d'afficher l'animation
    output_dir = r"C:\Users\leduc\OneDrive\Documents\École\Université\Session 6\PHS3903 - Projet III\Résultats"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "test3.mp4")

    print(f"Enregistrement de l'animation dans : {output_file}")
    anim.save(output_file, writer="ffmpeg", fps=60)

    plt.show() # We finally show the animation.

    ## Save the animation (Ubuntu).
    # anim.save('./animationsName.mp4', writer="ffmpeg", fps=60)

    return anim
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
    fig = plt.figure() # We create the figure.
    ax = fig.add_subplot(111, xlim=(0,L), ylim=(0,L)) # We add the subplot to the figure.

    img = ax.imshow(mod_psis[0], extent=[0,L,0,L], cmap=plt.get_cmap("hot"), vmin=0, vmax=np.max(mod_psis), zorder=1) # Here the modulus of the 2D wave function shall be represented.

    # We paint the walls of the double slit with rectangles.
    slitcolor = "w" # Color of the rectangles.
    slitalpha = 0.08 # Transparency of the rectangles.
    wall_bottom = Rectangle((j0*Dy,0),     w, i3*Dy,      color=slitcolor, zorder=50, alpha=slitalpha) # (x0, y0), width, height
    wall_middle = Rectangle((j0*Dy,i2*Dy), w, (i1-i2)*Dy, color=slitcolor, zorder=50, alpha=slitalpha)
    wall_top    = Rectangle((j0*Dy,i0*Dy), w, i3*Dy,      color=slitcolor, zorder=50, alpha=slitalpha)

    # We add the rectangular patches to the plot.
    ax.add_patch(wall_bottom)
    ax.add_patch(wall_middle)
    ax.add_patch(wall_top)

    # We define the animation function for FuncAnimation.

    def animate(i):
        
        img.set_data(mod_psis[i]) # Fill img with the modulus data of the wave function.
        img.set_zorder(1)
        
        return img, # We return the result ready to use with blit=True.


    anim = FuncAnimation(fig, animate, interval=1, frames =np.arange(0,Nt,2), repeat=False, blit=0) # We generate the animation.# Generamos la animación.

    plt.show() # We finally show the animation.

    # Save the animation (Ubuntu).
    anim.save('./animationsName.mp4', writer="ffmpeg", fps=60)


    return anim
####################################################

def saveData(mod_psis):
    """
    Enregistrer l'animation.

    Args : 
        mod_psis (plot) : Animation
    """
    
    # We transform the 3D array into a 2D array to save it with numpy.savetxt.
    mod_psis_reshaped = np.asarray(mod_psis).reshape(np.asarray(mod_psis).shape[0], -1) 
    
    # We save the 2D array as a text file.
    np.savetxt(r"C:\Users\leduc\OneDrive\Documents\École\Université\Session 6\PHS3903 - Projet III\Résultats", mod_psis_reshaped)
    
####################################################

def obtainData(Ny): 
    """
    Obtenir le data d'un vecteur de fonctions d'onde déjà créé.

    Args :
        Ny (int) : Grandeur du grillage en y.
    """
    
    # To obtain the data from the text file already created earlier.
    loaded_mod_psis = np.loadtxt("mod_psis_data.txt")
    
    # The loaded_mod_psis array is a 2D array, we need to return it to its original form.

    mod_psisshape2 = Ny-2

    # We finally obtain our mod_psis array.

    mod_psis = loaded_mod_psis.reshape( 
        loaded_mod_psis.shape[0], loaded_mod_psis.shape[1] // mod_psisshape2, mod_psisshape2) 
    
    ## For deleting the auxiliary 2D array.
    # del loaded_mod_psis
    
