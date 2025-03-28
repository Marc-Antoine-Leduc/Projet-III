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

def makeAnimationForSlits(mod_psis, v, L, Nt, extract_frac=0.75):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import os

    fig, ax = plt.subplots()
    
    img_wave = ax.imshow(mod_psis[0]**2, extent=[0, L, 0, L], origin='lower',
                         cmap='hot', vmin=0, vmax=np.max(mod_psis[0]**2))
    
    img_pot = ax.imshow(v.real, extent=[0, L, 0, L], origin='lower',
                        cmap='gray', alpha=0.3,  
                        vmin=0, vmax=np.max(v.real))
    
    # Déterminer la position x_extract et tracer une ligne verticale
    x_extract = extract_frac * L
    ax.axvline(x=x_extract, color='cyan', linestyle='--', linewidth=2, label='Patron extrait')
    
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.legend()

    def update(frame):
        wave_sq = mod_psis[frame]**2
        img_wave.set_data(wave_sq)
        img_wave.set_clim(vmin=0, vmax=np.max(wave_sq))
        return (img_wave, img_pot)
    
    anim = FuncAnimation(fig, update, frames=Nt, interval=50, blit=False)
    
    plt.show()
    
    output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "basicAnimation.mp4")
    print(f"Enregistrement de l'animation dans : {output_file}")
    anim.save(output_file, writer="ffmpeg", fps=60)

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

def diffractionPatron(mod_psis, L, Ny, s, a, k, n0=0, extract_frac=0.75):
    """
    Affiche le patron de diffraction cumulé sur l'écran à x = extract_frac * L,
    en ne prenant en compte que les instants après n0, et superpose le patron théorique.

    Args:
        mod_psis (list of arrays): Liste des modules de la fonction d'onde à chaque pas de temps.
        L (float): Longueur du domaine.
        Ny (int): Nombre de points dans la direction y.
        s (float): Distance entre les fentes (défini dans potentiel.py).
        a (float): Largeur effective des fentes (défini dans potentiel.py).
        k (float): Vecteur d'onde.
        n0 (int, optionnel): Indice de temps à partir duquel commencer le cumul.
        extract_frac (float, optionnel): Fraction de L pour l'extraction (ex. 0.75).
    
    Returns:
        cumulative_intensity (array): Intensité cumulative sur l'écran (à x = extract_frac * L) après n0.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from doubleSlit_FPB_CN import theoreticalIntensity

    # Maillage spatial sur [0, L]
    y_screen = np.linspace(0, L, mod_psis[0].shape[0])
    
    # Déterminer l'indice de la colonne correspondant à x = extract_frac * L.
    j_extract = int(extract_frac * mod_psis[0].shape[1])
    
    # Cumul de l'intensité sur la colonne d'extraction, à partir de n0
    cumulative_intensity = np.zeros(mod_psis[0].shape[0])

    # n1 = len(mod_psis) * 0.75 à penser pour pas considérer réflections à droite

    for psi in mod_psis[n0:]:
        cumulative_intensity += np.abs(psi[:, j_extract])**2

    # Pour le patron théorique, on centre la coordonnée y autour de L/2
    y_centered = y_screen - L/2

    theo_intensity = theoreticalIntensity(y_centered, s, a, L, k)

    max_sim  = np.max(cumulative_intensity)
    max_theo = np.max(theo_intensity)
    if max_theo == 0:
        max_theo = 1e-15  # éviter division par zéro
    theo_intensity_norm = theo_intensity * (max_sim / max_theo)

    plt.figure(figsize=(8, 6))
    plt.plot(y_screen, cumulative_intensity, label='Patron simulé')
    plt.plot(y_screen, theo_intensity_norm, 'k--', label='Patron théorique')
    plt.xlabel('Position y')
    plt.ylabel('Intensité cumulative (|ψ|²)')
    plt.title("Comparaison du patron de diffraction simulé et théorique")
    plt.grid(True)
    plt.legend()
    plt.show()

    return cumulative_intensity



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
    