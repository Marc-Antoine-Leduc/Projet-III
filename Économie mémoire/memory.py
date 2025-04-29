import numpy as np
import psutil
from potentiel import *
from doubleSlit_FPB_CN import *
from createAnimations import *
from scipy.stats import linregress

def errorConv(mod_psis_ar):
    ref_psi = mod_psis_ar[0]
    erreur = []
        
    for i in range(1, len(mod_psis_ar)):
        psi_i = mod_psis_ar[i]

        # Dimensions de la référence et de la comparaison
        nx0, ny0 = ref_psi.shape
        nxi, nyi = psi_i.shape
        Lx = Ly = 20

        x0 = np.linspace(0, Lx, nx0)
        y0 = np.linspace(0, Ly, ny0)
        xi = np.linspace(0, Lx, nxi)
        yi = np.linspace(0, Ly, nyi)

        x_common = np.intersect1d(x0, xi)
        y_common = np.intersect1d(y0, yi)

        mask_x0 = np.isin(x0, x_common)
        mask_y0 = np.isin(y0, y_common)
        mask_xi = np.isin(xi, x_common)
        mask_yi = np.isin(yi, y_common)

        psi0_common = ref_psi[np.ix_(mask_x0, mask_y0)]
        psii_common = psi_i[np.ix_(mask_xi, mask_yi)]

        err_i = np.linalg.norm(psi0_common - psii_common) / np.linalg.norm(psii_common)
        erreur.append(err_i)

    erreur = np.array(erreur)
    d_ar = d_ar[1:]

    log_d = np.log(d_ar)
    log_err = np.log(erreur)

    slope, intercept, r_value, _, _ = linregress(log_d, log_err)
    fit = slope * log_d + intercept

    plt.plot(log_d, log_err, 'o', label='Erreur en échelle logarithmique')
    plt.plot(log_d, fit, '--', label=f"Régression linéaire\npente = {slope:.2f}")
    plt.xlabel("log(Pas d'espace)")
    plt.ylabel("log(Erreur de convergence)")
    plt.title("Erreur de convergence en fonction du pas d'espace sur une échelle log-log")
    plt.legend()
    plt.grid(True)
    plt.show()

    return

def memoryCalcul(d_ar, mem_ar):
    plt.loglog(d_ar[::-1],mem_ar[::-1]/1024.0**3,'-o')
    plt.title('Exigences de mémoire')
    plt.xlabel('Pas $d_x=d_y$ [m]')
    plt.ylabel('Mémoire [Gb]')
    plt.show()   

    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "mémoire.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    return