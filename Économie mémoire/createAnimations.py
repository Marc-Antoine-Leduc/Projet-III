import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from doubleSlit_FPB_CN import theoreticalIntensity
from scipy.signal import argrelextrema, savgol_filter

def makeAnimationForSlits(mod_psis, v, L, Nt, n0, v_g, Dt, x0, j0, j1, i0, i1, i2, i3, w, Dy, extract_frac, x_fentes, x_extract, D, sigma):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle
    import os

    # Position à partir de laquelle psi est pris en compte (x = x0 + v_g * t)
    x_n0 = x0 + v_g * n0 * Dt  # Position à t = n0 * Dt

    fig, ax = plt.subplots()

    # Afficher la fonction d'onde
    img_wave = ax.imshow(mod_psis[0]**2, extent=[0, L, 0, L], origin='lower',
                         cmap='hot', vmin=0, vmax=np.max(mod_psis[0]**2))
    
    # Dessiner les fentes comme des rectangles
    slitcolor = "gray"  # Couleur des murs
    slitalpha = 0.8     # Transparence des murs
    wall_bottom = Rectangle((j0*Dy, 0), w, i0*Dy, color=slitcolor, zorder=50, alpha=slitalpha)  # Bas
    wall_middle = Rectangle((j0*Dy, i1*Dy), w, (i2-i1)*Dy, color=slitcolor, zorder=50, alpha=slitalpha)  # Milieu
    wall_top = Rectangle((j0*Dy, i3*Dy), w, (L-i3*Dy), color=slitcolor, zorder=50, alpha=slitalpha)  # Haut

    ax.add_patch(wall_bottom)
    ax.add_patch(wall_middle)
    ax.add_patch(wall_top)

    # Calcul de la position limite imposée par T
    x_T = x0 + v_g * (Nt * Dt)  # Position à t = T

    # Ligne verticale pour x_T (limite imposée par T)
    ax.axvline(x=x_T, color='purple', linestyle=':', linewidth=2, label=f'Limite T (x={x_T:.2f})')
    
    # Ligne verticale pour x_extract (position de l'écran)
    ax.axvline(x=x_extract, color='cyan', linestyle='--', linewidth=2, label='Patron extrait')
    
    # Ligne verticale pour x_n0 (début de la prise en compte de psi dans le cumul)
    ax.axvline(x=x_n0, color='green', linestyle='-.', linewidth=2, 
               label=f'Début cumul (n0={n0}, x={x_n0:.2f})')
    
    # Ligne horizontale pour D, au bas de l'animation (y = 0)
    ax.hlines(y=0, xmin=x_fentes, xmax=x_extract, colors='yellow', linestyles='-', linewidth=5, 
              label=f'Distance D = {D:.2f}', zorder=100)
    
    # Ajouter une étiquette au milieu de la ligne D
    ax.text((x_fentes + x_extract) / 2, 0.5, f'D = {D:.2f}', color='yellow', fontsize=10, 
            ha='center', va='bottom', zorder=101)

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.legend()

    max_psi = np.max(mod_psis[0]**2)  # Intensité max du paquet initial
    img_wave = ax.imshow(mod_psis[0]**2, extent=[0, L, 0, L], origin='lower',
                         cmap='hot', vmin=0, vmax=max_psi)

    def update(frame):
        wave_sq = mod_psis[frame]**2
        img_wave.set_data(wave_sq)
        img_wave.set_clim(vmin=0, vmax=max_psi)
        return (img_wave,)
    
    anim = FuncAnimation(fig, update, frames=Nt, interval=50, blit=False)
    
    plt.show()
    
    output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "basicAnimation.mp4")
    print(f"Enregistrement de l'animation dans : {output_file}")
    anim.save(output_file, writer="ffmpeg", fps=60)

    return anim

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

def fit(y_screen, cumulative_intensity, s, a_initial, k, D, L):
    """
    Effectue un curve fitting pour extraire la largeur des fentes (a) à partir du patron de diffraction simulé.

    Args:
        y_screen (array): Coordonnées y sur l'écran (de 0 à L).
        cumulative_intensity (array): Intensité cumulée simulée (|ψ|²).
        s (float): Distance entre les fentes (définie dans potentiel.py).
        a_initial (float): Estimation initiale de la largeur des fentes (pour l'optimisation).
        k (float): Vecteur d'onde.
        D (float): Distance entre le plan des fentes et l'écran.
        L (float): Longueur du domaine de simulation.

    Returns:
        a_fit (float): Largeur des fentes ajustée.
        I_0_fit (float): Facteur d'échelle de l'intensité ajusté.
    """
    from scipy.optimize import curve_fit

    # Définir la fonction à ajuster
    def intensity_to_fit(y, a, I_0):
        """
        Fonction d'intensité théorique avec a et I_0 comme paramètres à ajuster.
        """
        y_centered = y - L/2  # Centrer les coordonnées y autour de L/2
        return theoreticalIntensity(y_centered, s, a, D, k, I_0=I_0)

    # Normaliser l'intensité simulée pour faciliter l'ajustement
    max_sim = np.max(cumulative_intensity)
    if max_sim == 0:
        max_sim = 1e-15  # Éviter division par zéro
    normalized_intensity = cumulative_intensity / max_sim

    # Estimation initiale des paramètres [a, I_0]
    p0 = [a_initial, 1.0]  # a_initial est la valeur théorique, I_0 initial = 1

    try:
        popt, pcov = curve_fit(intensity_to_fit, y_screen, normalized_intensity, p0=p0, bounds=([0, 0], [np.inf, np.inf]))
        a_fit, I_0_fit = popt
        perr = np.sqrt(np.diag(pcov))  # Erreur standard sur les paramètres
        a_err, I_0_err = perr
    except RuntimeError as e:
        print(f"Erreur lors de l'ajustement : {e}")
        a_fit, I_0_fit = a_initial, 1.0
        a_err, I_0_err = 0, 0

    # Calculer les patrons pour comparaison
    y_centered = y_screen - L/2
    theo_intensity_initial = theoreticalIntensity(y_centered, s, a_initial, D, k, I_0=1.0)
    theo_intensity_fit = theoreticalIntensity(y_centered, s, a_fit, D, k, I_0=I_0_fit)

    # Renormaliser pour le tracé
    theo_intensity_initial = theo_intensity_initial * max_sim
    theo_intensity_fit = theo_intensity_fit * max_sim

    print(f"Largeur des fentes ajustée (a) : {a_fit:.4f} ± {a_err:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(y_screen, cumulative_intensity, label='Patron simulé', color='blue')
    plt.plot(y_screen, theo_intensity_initial, 'k--', label='Patron théorique initial', alpha=0.7)
    plt.plot(y_screen, theo_intensity_fit, 'r-', label='Patron ajusté', alpha=0.7)
    plt.xlabel('Position y')
    plt.ylabel('Intensité cumulative (|ψ|²)')
    plt.title("Ajustement du patron de diffraction pour extraire la largeur des fentes")
    plt.grid(True)
    plt.legend()
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "diffraction_fit.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()

    return a_fit, I_0_fit

def diffractionPatron(mod_psis, L, Ny, s, a, k, D, extract_frac=0.85, x0=None, Dt=None, v_g=None):
    """
    Affiche le patron de diffraction cumulé sur l'écran à x = extract_frac * L,
    en ne prenant en compte que les instants après n0, jusqu'à ce que x_extract soit atteint.
    Calcule et affiche la vitesse de propagation expérimentale.
    
    Args:
        mod_psis (list of arrays): Liste des modules de la fonction d'onde à chaque pas de temps.
        L (float): Longueur du domaine.
        Ny (int): Nombre de points dans la direction y.
        s (float): Distance entre les fentes.
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
    x_extract = extract_frac * len(mod_psis)
    
    # Trouver le pas de temps où x_extract est atteint
    n_stop = len(mod_psis)  # Par défaut, utiliser tous les pas de temps

    # Calculer l'intensité maximale à x_extract
    max_intensity_at_extract = np.max([np.sum(np.abs(psi[:, j_extract])**2) for psi in mod_psis])
    threshold = 0.03 * max_intensity_at_extract  # Sert à détecter le moment où le paquet d'onde atteint 𝑥_extract, car le paquet d'onde semble ralenti

    # Quand l'intensité de l'onde à 𝑥_extract devient assez grande, on considère que le paquet est arrivé. Permet de corriger v_g
        
    for n, psi in enumerate(mod_psis):
        intensity_at_extract = np.sum(np.abs(psi[:, j_extract])**2)
        if intensity_at_extract > threshold:
            n_stop = n + 10  # Ajouter quelques pas pour capturer le passage complet
            break

    print(f"x_extract atteint à n_stop = {n_stop}, len(mod_psis) = {len(mod_psis)}")
    
    # S'assurer que n_stop ne dépasse pas Nt
    n_stop = min(n_stop, len(mod_psis))
    
    # Diagnostic pour vérifier si threshold est approprié
    # Calculer l'intensité à x_extract pour chaque pas de temps à partir de n0
    intensities = [np.sum(np.abs(psi[:, j_extract])**2) for psi in mod_psis[:n_stop]]

    # Calculer les temps correspondants
    times = np.arange(0, n_stop) * Dt

    plt.figure(figsize=(10, 6))
    plt.plot(times, intensities, label='Intensité à x_extract')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Seuil = {threshold:.2e}')
    plt.axvline(x=(n_stop - 0) * Dt + 0 * Dt, color='g', linestyle='--', label=f'n_stop = {n_stop}')
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
    elif n_stop - 0 < 10:
        print("Avertissement : n_stop est très proche de n0. Le seuil est probablement trop bas.")
    else:
        print(f"n_stop détecté à t = {(n_stop - 0) * Dt + 0 * Dt:.4f} s")
    
    # Calculer la vitesse expérimentale 
    if x0 is not None and Dt is not None and v_g is not None:
        if n_stop > 0:
            time_to_extract = (n_stop - 0) * Dt  # Temps écoulé depuis n0
            distance = x_extract - x0
            v_exp = distance / time_to_extract if time_to_extract > 0 else 0
            print(f"Vitesse expérimentale : v_exp = {v_exp:.4f}")
            print(f"Vitesse théorique : v_g = {v_g:.4f}")
            print(f"Différence relative : {abs(v_exp - v_g) / v_g * 100:.2f}%")
        else:
            print("Erreur : x_extract n'a pas été atteint après n0.")
    
    # Cumul de l'intensité sur la colonne d'extraction, de n0 à n_stop
    cumulative_intensity = np.zeros(mod_psis[0].shape[0])
    for psi in mod_psis[:n_stop]:
        cumulative_intensity += np.abs(psi[:, j_extract])**2

    # Pour le patron théorique, on centre la coordonnée y autour de L/2
    y_centered = y_screen - L/2
    theo_intensity = theoreticalIntensity(y_centered, s, a, D, k)
    max_sim = np.max(cumulative_intensity)
    max_theo = np.max(theo_intensity)
    if max_theo == 0:
        max_theo = 1e-15  # éviter division par zéro
    theo_intensity_norm = theo_intensity * (max_sim / max_theo)

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
        print(f"Largeur des fentes théorique : {a:.4f}")
    else:
        y_left, y_right = None, None
        a_calculated = None
        print("Impossible de détecter les minimas.")

    plt.figure(figsize=(10, 6))
    plt.plot(y_screen, cumulative_intensity, label='Patron simulé')
    plt.plot(y_screen, theo_intensity_norm, 'k--', label='Patron théorique')
    plt.plot(y_screen, smoothed_intensity, 'm-', alpha=0.5, label='Enveloppe lissée') 
    if y_left is not None and y_right is not None:
        plt.axvline(x=y_left, color='r', linestyle='--', label=f'Minima gauche (y={y_left:.2f})')
        plt.axvline(x=y_right, color='g', linestyle='--', label=f'Minima droite (y={y_right:.2f})')
    plt.xlabel('Position y')
    plt.ylabel('Intensité cumulative (|ψ|²)')
    plt.title("Comparaison du patron de diffraction simulé et théorique")
    plt.grid(True)
    plt.legend()
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "diffraction_patron.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()

    a_fit, I_0_fit = fit(y_screen, cumulative_intensity, s, a, k, D, L)

    return cumulative_intensity
