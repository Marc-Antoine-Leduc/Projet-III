import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.signal import argrelextrema, savgol_filter

def makeAnimationForSlits(
    mod_psis, v, L, Nt, n0, v_g, Dt,
    x0, Dy, extract_frac, x_fentes, x_extract, D, sigma
):
    """
    Animation de la diffraction d'une chaîne 1D d'atomes.

    Args:
        mod_psis      (list of 2D arrays): modules de ψ à chaque pas de temps.
        v             (2D array):            potentiel périodique (non affiché ici).
        L             (float):               taille du domaine.
        Nt            (int):                 nombre total de pas de temps.
        n0            (int):                 pas de temps de début de cumul.
        v_g           (float):               vitesse de groupe.
        Dt            (float):               pas de temps.
        x0            (float):               position initiale du paquet.
        Dy            (float):               pas d'espace.
        extract_frac  (float):               fraction de L pour l'écran.
        x_fentes      (float):               position du plan atomique (L/2).
        x_extract     (float):               position de l'écran.
        D             (float):               distance chaîne→écran.
        sigma         (float):               largeur du paquet (non utilisé ici).
    Returns:
        anim (FuncAnimation): l’animation matplotlib.
    """
    # Calculs de positions
    x_n0 = x0 + v_g * n0 * Dt       # où commence le cumul
    x_T  = x0 + v_g * Nt  * Dt      # fin de simulation en x

    # Préparation de la figure
    fig, ax = plt.subplots(figsize=(6,6))

    # Affichage initial de |ψ|², sans bande grise
    max_psi = np.max(mod_psis[0]**2)
    img = ax.imshow(
        mod_psis[0]**2,
        extent=[Dy/2, L-Dy/2, Dy/2, L-Dy/2],
        origin='lower',
        cmap='hot',
        vmin=0, vmax=max_psi
    )

    # Verticales :
    ax.axvline(x=x_T,       color='purple', linestyle=':',  lw=2,
               label=f'Limite T (x={x_T:.1f})')
    ax.axvline(x=x_extract, color='cyan',   linestyle='--', lw=2,
               label=f'Patron extrait (x={x_extract:.1f})')
    ax.axvline(x=x_n0,      color='green',  linestyle='-.', lw=2,
               label=f'Début cumul (x={x_n0:.1f})')
    ax.axvline(x=x_fentes,  color='white',  linestyle='--', lw=2,
               label=f'Plan atomique (x={x_fentes:.1f})')

    # Ligne horizontale pour D
    y0_line = Dy/2
    ax.hlines(y=y0_line, xmin=x_fentes, xmax=x_extract,
              colors='yellow', linestyles='-', lw=5,
              label=f'Distance D = {D:.1f}')
    ax.text((x_fentes + x_extract)/2, y0_line*1.5,
            f'D = {D:.1f}', color='yellow',
            ha='center', va='bottom')

    ax.set_xlim(Dy/2,   L-Dy/2)
    ax.set_ylim(Dy/2,   L-Dy/2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("Diffraction d'une chaîne 1D d'atomes")
    ax.legend(loc='upper right')

    # Fonction de mise à jour
    def update(frame):
        img.set_data(mod_psis[frame]**2)
        img.set_clim(0, max_psi)
        return (img,)

    anim = FuncAnimation(fig, update, frames=Nt, interval=50, blit=False)

    plt.show()

    # Enregistrement automatique
    os.makedirs('.', exist_ok=True)
    out = os.path.join('.', 'chain_diffraction.mp4')
    print(f"Saving animation to {out}")
    anim.save(out, writer='ffmpeg', fps=60)

    return anim

def diffractionPatron(mod_psis, L, Ny, a, k, D, n0=0, extract_frac=0.75, x0=None, Dt=None, v_g=None):
    """
    Affiche le patron de diffraction cumulé sur l'écran à x = extract_frac * L,
    en ne prenant en compte que les instants après n0, jusqu'à ce que x_extract soit atteint.
    Calcule et affiche la vitesse de propagation expérimentale.
    
    Args:
        mod_psis (list of arrays): Liste des modules de la fonction d'onde à chaque pas de temps.
        L (float): Longueur du domaine.
        Ny (int): Nombre de points dans la direction y.
        a (float): Largeur effective des fentes.
        k (float): Vecteur d'onde.
        D (float): Distance entre le plan des fentes et l'écran.
        n0 (int, optional): Indice de temps à partir duquel commencer le cumul.
        extract_frac (float, optional): Fraction de L pour l'extraction (ex. 0.75).
        x0 (float, optional): Position initiale du paquet, pour calculer la vitesse expérimentale.
        Dt (float, optional): Pas de temps, pour calculer la vitesse expérimentale.
        v_g (float, optional): Vitesse de groupe théorique, pour comparaison.
    
    Returns:
        cumulative_intensity (array): Intensité cumulative sur l'écran.
    """
    # Maillage spatial sur [0, L]
    y_screen = np.linspace(0, L, mod_psis[0].shape[0])
    
    # Déterminer l'indice de la colonne correspondant à x = extract_frac * L
    j_extract = int(extract_frac * mod_psis[0].shape[1])
    x_extract = extract_frac * L
    
    # Trouver le pas de temps où x_extract est atteint
    n_stop = len(mod_psis)  # Par défaut, utiliser tous les pas de temps

    # Calculer l'intensité maximale à x_extract
    max_intensity_at_extract = np.max([np.sum(np.abs(psi[:, j_extract])**2) for psi in mod_psis])
    threshold = 0.03 * max_intensity_at_extract  # Sert à détecter le moment où le paquet d'onde atteint 𝑥_extract, car le paquet d'onde semble ralenti

    # Quand l'intensité de l'onde à 𝑥_extract devient assez grande, on considère que le paquet est arrivé. Permet de corriger v_g
        
    for n, psi in enumerate(mod_psis[n0:], start=n0):
        intensity_at_extract = np.sum(np.abs(psi[:, j_extract])**2)
        if intensity_at_extract > threshold:
            n_stop = n + 10  # Ajouter quelques pas pour capturer le passage complet
            break

    print(f"x_extract atteint à n_stop = {n_stop}, len(mod_psis) = {len(mod_psis)}")
    
    # S'assurer que n_stop ne dépasse pas Nt
    n_stop = min(n_stop, len(mod_psis))
    
    # Diagnostic pour vérifier si threshold est approprié
    # Calculer l'intensité à x_extract pour chaque pas de temps à partir de n0
    intensities = [np.sum(np.abs(psi[:, j_extract])**2) for psi in mod_psis[n0:n_stop]]

    # Calculer les temps correspondants
    times = np.arange(n0, n_stop) * Dt

    plt.figure(figsize=(10, 6))
    plt.plot(times, intensities, label='Intensité à x_extract')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Seuil = {threshold:.2e}')
    plt.axvline(x=(n_stop - n0) * Dt + n0 * Dt, color='g', linestyle='--', label=f'n_stop = {n_stop}')
    if x0 is not None and v_g is not None:
        t_theoretical = (x_extract - x0) / v_g
        n_theoretical = int(t_theoretical / Dt)
        plt.axvline(x=t_theoretical, color='b', linestyle='--', label=f'Temps théorique = {t_theoretical:.4f} s (n={n_theoretical})')
    plt.xlabel('Temps (s)')
    plt.ylabel('Intensité (|ψ|²)')
    plt.title("Évolution de l'intensité à x_extract")
    plt.grid(True)
    plt.legend()
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "intensity_at_x_extract.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()

    if n_stop == len(mod_psis):
        print("Avertissement : n_stop a atteint la fin de la simulation. Le seuil est probablement trop élevé.")
    elif n_stop - n0 < 10:
        print("Avertissement : n_stop est très proche de n0. Le seuil est probablement trop bas.")
    else:
        print(f"n_stop détecté à t = {(n_stop - n0) * Dt + n0 * Dt:.4f} s")
    
    # Calculer la vitesse expérimentale 
    if x0 is not None and Dt is not None and v_g is not None:
        if n_stop > n0:
            time_to_extract = (n_stop - n0) * Dt  # Temps écoulé depuis n0
            distance = x_extract - x0
            v_exp = distance / time_to_extract if time_to_extract > 0 else 0
            print(f"Vitesse expérimentale : v_exp = {v_exp:.4f}")
            print(f"Vitesse théorique : v_g = {v_g:.4f}")
            print(f"Différence relative : {abs(v_exp - v_g) / v_g * 100:.2f}%")
        else:
            print("Erreur : x_extract n'a pas été atteint après n0.")
    
    # Cumul de l'intensité sur la colonne d'extraction, de n0 à n_stop
    cumulative_intensity = np.zeros(mod_psis[0].shape[0])
    for psi in mod_psis[n0:n_stop]:
        cumulative_intensity += np.abs(psi[:, j_extract])**2

    # Pour le patron théorique, on centre la coordonnée y autour de L/2
    y_centered = y_screen - L/2
    #theo_intensity = theoreticalIntensity(y_centered, s, a, D, k)
    max_sim = np.max(cumulative_intensity)
    #max_theo = np.max(theo_intensity)
    # if max_theo == 0:
    #     max_theo = 1e-15  # éviter division par zéro
    #theo_intensity_norm = theo_intensity * (max_sim / max_theo)

    # --- Extraction de la largeur des fentes ---
    lambda_ = 2 * np.pi / k  # Longueur d'onde

    # Lisser le patron simulé pour isoler l'enveloppe
    window_size = 11 
    if window_size >= len(cumulative_intensity):
        window_size = len(cumulative_intensity) // 2 * 2 + 1 
    smoothed_intensity = savgol_filter(cumulative_intensity, window_size, 3) # Lissage 

    # Trouver les minimas locaux dans l'enveloppe lissée
    center_idx = len(y_screen) // 2
    minima_indices = argrelextrema(smoothed_intensity, np.less)[0]  # Indices des minimas locaux

    # Séparer les minimas à gauche et à droite du centre
    left_minima = minima_indices[minima_indices < center_idx]
    right_minima = minima_indices[minima_indices > center_idx]

    # Prendre le dernier minima à gauche et le premier minima à droite les plus proches du centre
    if len(left_minima) > 0 and len(right_minima) > 0:
        y_left = y_screen[left_minima[-1]]  # Dernier minima à gauche
        y_right = y_screen[right_minima[0]]  # Premier minima à droite
        delta_y = y_right - y_left

        # Calcul de la largeur des fentes
        a_calculated = (2 * lambda_ * D) / delta_y  # y_n = nλD/a

        print(f"Largeur des fentes calculée : {a_calculated:.4f}")
        #print(f"Largeur des fentes théorique : {a:.4f}")
    else:
        y_left, y_right = None, None
        a_calculated = None
        print("Impossible de détecter les minimas.")

    # plt.figure(figsize=(10, 6))
    # plt.plot(y_screen, cumulative_intensity, label='Patron simulé')
    # plt.plot(y_screen, theo_intensity_norm, 'k--', label='Patron théorique')
    # plt.plot(y_screen, smoothed_intensity, 'm-', alpha=0.5, label='Enveloppe lissée') 
    # if y_left is not None and y_right is not None:
    #     plt.axvline(x=y_left, color='r', linestyle='--', label=f'Minima gauche (y={y_left:.2f})')
    #     plt.axvline(x=y_right, color='g', linestyle='--', label=f'Minima droite (y={y_right:.2f})')
    # plt.xlabel('Position y')
    # plt.ylabel('Intensité cumulative (|ψ|²)')
    # plt.title("Comparaison du patron de diffraction simulé et théorique")
    # plt.grid(True)
    # plt.legend()
    # output_dir = "figures"
    # os.makedirs(output_dir, exist_ok=True)
    # output_file = os.path.join(output_dir, "diffraction_patron.png")
    # plt.savefig(output_file, dpi=150, bbox_inches='tight')
    # plt.show()

    # a_fit, I_0_fit = fit(y_screen, cumulative_intensity, s, a, k, D, L)

    plt.figure(figsize=(10, 6))
    plt.plot(y_screen, cumulative_intensity, label='Patron simulé')
    plt.xlabel('Position y')
    plt.ylabel('Intensité cumulative (|ψ|²)')
    plt.title('Patron de diffraction simulé')
    plt.grid(True)
    plt.legend()

    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "diffraction_patron.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
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
    