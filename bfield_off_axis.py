import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# ── analytical on-axis expression ────────────────────────────────────────────
def beta_z_onaxis(zeta, eta):
    t1 = (0.5 - zeta) / np.sqrt(eta**2 + (0.5 - zeta)**2)
    t2 = (0.5 + zeta) / np.sqrt(eta**2 + (0.5 + zeta)**2)
    return 0.5 * (t1 + t2)

# ── dimensionless vector potential ───────────────────────────────────────────
def A_tilde(sigma, zeta, eta):
    def integrand(phi_prime, zeta_prime):
        denom = np.sqrt(sigma**2 + eta**2
                        - 2*sigma*eta*np.cos(phi_prime)
                        + (zeta - zeta_prime)**2)
        return np.cos(phi_prime) / denom

    result, _ = integrate.dblquad(
        integrand,
        -0.5, 0.5,
        0, 2*np.pi,
        epsabs=1e-6, epsrel=1e-6
    )
    return (eta / (4*np.pi)) * result

# ── compute field on half-plane sigma >= 0, then mirror ─────────────────────
def compute_B_field(eta, n_sigma=25, n_zeta=40, sigma_max=2.0, zeta_max=2.0):
    sigma_pos = np.linspace(0.0, sigma_max, n_sigma)
    zeta_vals = np.linspace(-zeta_max, zeta_max, n_zeta)

    A_pos = np.zeros((n_sigma, n_zeta))

    print(f"Computing A_phi grid for eta={eta} ...")
    for i, s in enumerate(sigma_pos):
        if s == 0.0:
            A_pos[i, :] = 0.0
        else:
            for j, z in enumerate(zeta_vals):
                A_pos[i, j] = A_tilde(s, z, eta)
        print(f"  row {i+1}/{n_sigma}  (sigma={s:.3f})")

    d_zeta = zeta_vals[1] - zeta_vals[0]

    # beta_z = (1/sigma) * d(sigma*A)/d(sigma)
    sigma_A    = sigma_pos[:, None] * A_pos
    denom      = np.where(sigma_pos[:, None] == 0, 1.0, sigma_pos[:, None])
    beta_z_pos = np.gradient(sigma_A, sigma_pos, axis=0) / denom
    beta_z_pos[0, :] = beta_z_onaxis(zeta_vals, eta)

    # beta_s = -dA/dzeta
    beta_s_pos       = -np.gradient(A_pos, d_zeta, axis=1)
    beta_s_pos[0, :] = 0.0

    # ── interior mask: uniform field inside magnet ────────────────────────────
    # physically: B = mu0*M inside a uniformly magnetized cylinder
    # so beta_z = 1, beta_s = 0 everywhere inside (sigma <= eta, |zeta| <= 0.5)
    for i, s in enumerate(sigma_pos):
        for j, z in enumerate(zeta_vals):
            if s <= eta and abs(z) <= 0.5:
                beta_z_pos[i, j] = 1.0
                beta_s_pos[i, j] = 0.0

    # ── mirror to negative sigma ──────────────────────────────────────────────
    sigma_neg   = -sigma_pos[1:][::-1]
    beta_z_neg  =  beta_z_pos[1:][::-1]   # B_z even in sigma
    beta_s_neg  = -beta_s_pos[1:][::-1]   # B_s odd in sigma

    sigma_full  = np.concatenate([sigma_neg, sigma_pos])
    beta_z_full = np.concatenate([beta_z_neg, beta_z_pos], axis=0)
    beta_s_full = np.concatenate([beta_s_neg, beta_s_pos], axis=0)

    return sigma_full, zeta_vals, beta_s_full, beta_z_full

# ── plot ─────────────────────────────────────────────────────────────────────
def plot_field(sigma_vals, zeta_vals, beta_s, beta_z, eta):
    B_mag = np.sqrt(beta_s**2 + beta_z**2)

    beta_z_p = beta_z.T
    beta_s_p = beta_s.T
    B_mag_p  = B_mag.T

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    datasets = [
        (B_mag_p,  r'$|\mathbf{B}|/\mu_0 M$', 'viridis', 'Field magnitude'),
        (beta_z_p, r'$B_z / \mu_0 M$',         'RdBu_r',  'Axial component'),
    ]

    for ax, (data, cblabel, cmap, title) in zip(axes, datasets):
        cf = ax.contourf(sigma_vals, zeta_vals, data, levels=60, cmap=cmap)
        plt.colorbar(cf, ax=ax, label=cblabel, fraction=0.046, pad=0.04)

        ax.streamplot(
            sigma_vals, zeta_vals,
            beta_s_p, beta_z_p,
            color='white', linewidth=0.7, density=1.2, arrowsize=0.7
        )

        ax.add_patch(plt.Rectangle(
            (-eta, -0.5), 2*eta, 1.0,
            edgecolor='red', facecolor='none', linewidth=2, label='Magnet'
        ))

        ax.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_xlabel(r'$\sigma = s/L$', fontsize=13)
        ax.set_ylabel(r'$\zeta = z/L$',  fontsize=13)
        ax.set_title(rf'{title}  ($\eta = R/L = {eta}$)', fontsize=12)
        ax.legend(fontsize=10, loc='upper right')

    plt.tight_layout()
    fname = f'C:/Users/dsbor/OneDrive/Desktop/Personal/DIY-NMR-Spectrometer/bfield_2d_eta{eta}.png'
    plt.savefig(fname, dpi=150)
    print(f"Saved {fname}")
    plt.show()

# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    eta = 1.0

    sigma_vals, zeta_vals, beta_s, beta_z = compute_B_field(
        eta, n_sigma=25, n_zeta=40, sigma_max=2.0, zeta_max=2.0
    )
    plot_field(sigma_vals, zeta_vals, beta_s, beta_z, eta)