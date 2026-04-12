import numpy as np
import matplotlib.pyplot as plt

# dimensionless B field for a cylindrical permanent magnet
def beta_z(zeta: np.ndarray, eta: float) -> np.ndarray:
    term1 = (0.5 - zeta) / np.sqrt(eta**2 + (0.5 - zeta)**2)     # upper portion of magnet
    term2 = (0.5 + zeta) / np.sqrt(eta**2 + (0.5 + zeta)**2)     # lower portion of magnet

    return 0.5 * (term1 + term2)

zeta = np.linspace(-5, 5, 1000)
etas = [0.5, 1.0, 2.0]

fig, ax = plt.subplots(figsize=(8, 5))

for eta in etas:
    ax.plot(zeta, beta_z(zeta, eta), label=f"R/L = {eta}")

# shade magnet region
ax.axvspan(-0.5, 0.5, alpha=0.15, color="gray", label="Magnet Interior")
ax.axvline(-0.5, color="gray", linestyle="--", linewidth=0.8)
ax.axvline( 0.5, color="gray", linestyle="--", linewidth=0.8)

ax.set_xlabel(r"$\zeta = z/L$", fontsize=13)
ax.set_ylabel(r"$\beta = \frac{B_z}{\mu_0 M}$", fontsize=13)
ax.set_title(r'On-axis field of a uniformly magnetized cylinder', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('C:/Users/dsbor/OneDrive/Desktop/Personal/DIY-NMR-Spectrometer/bfield_cylinder.png', dpi=150)
plt.show()