# img = mpimg.imread(r"C:\Users\leduc\OneDrive\Documents\École\Université\Session 6\PHS3903 - Projet III\Évaluations finales\temps.png")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Charger l'image
img = mpimg.imread(r"C:\Users\leduc\OneDrive\Documents\École\Université\Session 6\PHS3903 - Projet III\Évaluations finales\temps.png")

# Afficher pour calibration
fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title('Clique sur 4 points de calibration: x_min, x_max, y_min, y_max (dans cet ordre)')
plt.axis('on')
plt.show(block=True)  # <-- IMPORTANT pour bien afficher

# Calibration
calib_points = plt.ginput(n=4, timeout=0)
plt.close()

# Points de calibration
pixel_x_min, pixel_y_xaxis = calib_points[0]
pixel_x_max, _ = calib_points[1]
pixel_y_min, pixel_x_yaxis = calib_points[2]
pixel_y_max, _ = calib_points[3]

# Valeurs connues physiquement
x_min = 0.03
x_max = 0.08
y_max_value = 10    # Bas
y_min_value = 850   # Haut

print("Calibration terminée. Maintenant, clique sur les points de données.")

# Maintenant nouvelle fenêtre pour cliquer sur les points de données
fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title('Clique sur tous les points de données (puis ferme)')
plt.axis('on')
plt.show(block=True)  # <-- IMPORTANT pour bloquer et attendre

data_points = plt.ginput(n=-1, timeout=0)
plt.close()

# Conversion pixel -> valeur physique
pixels_x = np.array([p[0] for p in data_points])
pixels_y = np.array([p[1] for p in data_points])

delta_x = x_min + (pixels_x - pixel_x_min) / (pixel_x_max - pixel_x_min) * (x_max - x_min)
time = y_min_value + (pixels_y - pixel_y_min) / (pixel_y_max - pixel_y_min) * (y_max_value - y_min_value)

# Afficher les données extraites
print("\nPoints extraits :")
for dx, t in zip(delta_x, time):
    print(f"Δx = {dx:.5f}, temps = {t:.2f} s")

# Régression log-log
log_dx = np.log(delta_x)
log_time = np.log(time)
pente, intercept = np.polyfit(log_dx, log_time, 1)

# Fit pour tracer
delta_x_fit = np.linspace(min(delta_x), max(delta_x), 100)
time_fit = np.exp(intercept) * delta_x_fit**pente

# Tracé final
plt.figure(figsize=(8,6))
plt.loglog(delta_x, time, 'o', label='Données extraites', markersize=8)
plt.loglog(delta_x_fit, time_fit, '--r', label=f"Fit: y ∝ x^{pente:.2f}")
plt.xlabel("Pas d'espace Δx (log)", fontsize=14)
plt.ylabel("Temps de simulation (s) (log)", fontsize=14)
plt.grid(True, which="both", ls="--")
plt.legend(fontsize=12)
plt.title("Temps de simulation vs Pas d'espace (log-log)", fontsize=16)
plt.tight_layout()
plt.show()

print(f"\nL'ordre d'augmentation estimé est : {abs(pente):.2f}")
