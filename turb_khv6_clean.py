# %%
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

#Configuration Settings#
# %%
@dataclass(slots=True)
class KHConfig:
  #set domain
    nx: int = 192
    ny: int = 192
    lx: float = 14.0
    ly: float = 14.0
    tmax: float = 100.0
    dt: float = 2e-3
  #set initial conditions
    Ri: float = 0.15
    Re: float = 1000.0
    Pr: float = 1.0
    U0: float = 1.0
    shear_thickness: float = 0.5
    density_thickness: float = 0.5
  #define perturbation conditions
    perturbation_amplitude: float = 1e-3
    perturbation_mode: int = 2
    perturbation_phase: float = 0.2
    noise_amplitude: float = 0.0
    seed: int = 0
  #set snapshot conditions/file outputs
    snapshot_every: int = 250
    outdir: str = "kh_bufferfly_raw"
    label: str = "run"
    verbose: bool = True

    @property
    def nu(self) -> float:
        return self.U0 * self.shear_thickness / self.Re

    @property
    def kappa(self) -> float:
        return self.nu / self.Pr

    @property
    def nt(self) -> int:
        return int(np.ceil(self.tmax / self.dt))

#Spectral utilities#
# %%
def make_wavenumbers(nx: int, ny: int, lx: float, ly: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx / nx) #np.fft.fftfreq function generate frequencies and then convert frequencies to wavelengths
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    KX, KY = np.meshgrid(kx, ky, indexing="xy") #create 2D array
    K2 = KX**2 + KY**2 #need for laplacian transform
    return KX, KY, K2

#derivative of x_hat
def deriv_x_hat(f_hat: np.ndarray, KX: np.ndarray) -> np.ndarray:
    return 1j * KX * f_hat

#derivative of y_hat
def deriv_y_hat(f_hat: np.ndarray, KY: np.ndarray) -> np.ndarray:
    return 1j * KY * f_hat

#transform back to physical space
def to_real(f_hat: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(f_hat).real

#solve laplacian
def laplacian_hat(f_hat: np.ndarray, K2: np.ndarray) -> np.ndarray:
    return -K2 * f_hat


def solve_minus_laplacian(rhs_hat: np.ndarray, K2: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rhs_hat, dtype=complex)
    mask = K2 != 0.0 #make sure only nonzero values
    out[mask] = rhs_hat[mask] / K2[mask]
    return out

#create 2D Mesh
def dealias_mask(nx: int, ny: int, frac: float = 2 / 3) -> np.ndarray: #2/3 rule
    kx_idx = np.fft.fftfreq(nx) * nx
    ky_idx = np.fft.fftfreq(ny) * ny
    KXI, KYI = np.meshgrid(kx_idx, ky_idx, indexing="xy")
    return (np.abs(KXI) < frac * nx / 2) & (np.abs(KYI) < frac * ny / 2)


def apply_mask(f_hat: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return f_hat
    out = np.zeros_like(f_hat)
    out[mask] = f_hat[mask]
    return out


def jacobian_hat(
    psi_hat: np.ndarray,
    q_hat: np.ndarray,
    KX: np.ndarray,
    KY: np.ndarray,
    mask: Optional[np.ndarray],
) -> np.ndarray:
    psi_x = to_real(deriv_x_hat(psi_hat, KX))
    psi_y = to_real(deriv_y_hat(psi_hat, KY))
    q_x = to_real(deriv_x_hat(q_hat, KX))
    q_y = to_real(deriv_y_hat(q_hat, KY))
    return apply_mask(np.fft.fft2(psi_x * q_y - psi_y * q_x), mask)

#Initial conditions#
# %%
def initialize_fields(cfg: KHConfig):
    rng = np.random.default_rng(cfg.seed)

    x = np.linspace(0.0, cfg.lx, cfg.nx, endpoint=False)
    y = np.linspace(0.0, cfg.ly, cfg.ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")

    yc = cfg.ly / 2.0
    yy = Y - yc

    U_base = cfg.U0 * np.tanh(yy / cfg.shear_thickness)

    B0 = cfg.Ri * cfg.U0**2 / cfg.shear_thickness
    b_base = B0 * np.tanh(yy / cfg.density_thickness)

    kx0 = 2.0 * np.pi * cfg.perturbation_mode / cfg.lx
    envelope_width = 1.5 * cfg.density_thickness
    envelope = np.exp(-(yy / envelope_width) ** 2)
    eta = cfg.perturbation_amplitude * np.cos(kx0 * X + cfg.perturbation_phase) * envelope

    sech2 = 1.0 / np.cosh(yy / cfg.density_thickness) ** 2
    dbdy = (B0 / cfg.density_thickness) * sech2
    b = b_base - eta * dbdy

    if cfg.noise_amplitude > 0.0:
        noise_env = np.exp(-(yy / (2.0 * envelope_width)) ** 2)
        b += cfg.noise_amplitude * rng.standard_normal(size=b.shape) * noise_env

    psi = cfg.U0 * cfg.shear_thickness * np.log(np.cosh(yy / cfg.shear_thickness))
    psi += eta

    KX, KY, K2 = make_wavenumbers(cfg.nx, cfg.ny, cfg.lx, cfg.ly)
    psi_hat = np.fft.fft2(psi)
    omega = to_real(-laplacian_hat(psi_hat, K2))

    return X, Y, omega, b, U_base, b_base

#Diagnostics#
# %%
def velocity_from_vorticity(omega: np.ndarray, KX: np.ndarray, KY: np.ndarray, K2: np.ndarray):
    omega_hat = np.fft.fft2(omega)
    psi_hat = solve_minus_laplacian(omega_hat, K2)
    psi = to_real(psi_hat)
    u = to_real(deriv_y_hat(psi_hat, KY))
    w = -to_real(deriv_x_hat(psi_hat, KX))
    return psi, u, w

#TKE
def total_tke(u: np.ndarray, w: np.ndarray) -> float:
    return 0.5 * np.mean(u**2 + w**2)


def perturbation_ke(u: np.ndarray, w: np.ndarray) -> float:
    ubar = np.mean(u, axis=1, keepdims=True)
    up = u - ubar
    return 0.5 * np.mean(up**2 + w**2)

#Enstrophy
def enstrophy(omega: np.ndarray) -> float:
    return 0.5 * np.mean(omega**2)

#Vertical Velocity w
def vertical_velocity_rms(w: np.ndarray) -> float:
    return float(np.sqrt(np.mean(w**2)))

#Mixing Thickness
def mixing_thickness(b: np.ndarray, y: np.ndarray) -> float:
    bbar = np.mean(b, axis=1)
    dbdy = np.gradient(bbar, y)
    max_grad = np.max(np.abs(dbdy))
    if max_grad < 1e-14:
        return np.nan
    return float((np.max(bbar) - np.min(bbar)) / max_grad)


def estimate_transition_time(times: np.ndarray, pke: np.ndarray, factor: float = 10.0) -> float:
    if len(pke) == 0:
        return np.nan
    threshold = factor * max(pke[0], 1e-20)
    hits = np.where(pke >= threshold)[0]
    return float(times[hits[0]]) if len(hits) else np.nan


def fit_growth_rate(
    times: np.ndarray,
    pke: np.ndarray,
    min_factor: float = 3.0,
    max_factor: float = 30.0,
    min_points: int = 8,
):
    if len(times) != len(pke) or len(times) < min_points:
        return np.nan, np.nan, np.nan, None

    p0 = max(pke[0], 1e-20)
    idx = np.where((pke >= min_factor * p0) & (pke <= max_factor * p0))[0]

    if len(idx) < min_points:
        idx = np.arange(min(max(min_points, int(0.2 * len(times))), len(times)))

    if len(idx) < min_points:
        return np.nan, np.nan, np.nan, None

    t_fit = times[idx]
    y_fit = np.log(np.maximum(pke[idx], 1e-30))
    sigma, intercept = np.polyfit(t_fit, y_fit, 1)

    y_pred = sigma * t_fit + intercept
    ss_res = np.sum((y_fit - y_pred) ** 2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    r2 = np.nan if ss_tot <= 1e-16 else 1.0 - ss_res / ss_tot
    return float(sigma), float(intercept), float(r2), idx


def classify_regime(growth_max: float, peak_enstrophy: float, mix_ratio: float, transition_time: float) -> str:
    if np.isnan(growth_max):
        return "unknown"
    if growth_max < 2.0:
        return "stable"
    if growth_max < 10.0 and mix_ratio < 1.10:
        return "weak_KH"
    if growth_max >= 10.0 and mix_ratio < 1.30 and (np.isnan(transition_time) or transition_time > 1e8):
        return "KH_billows"
    if growth_max >= 10.0 and mix_ratio >= 1.30 and peak_enstrophy < 3.0:
        return "transitional"
    if growth_max >= 10.0 and mix_ratio >= 1.30:
        return "strong_breakdown"
    return "transitional"


def reynolds_stress_fields(u: np.ndarray, w: np.ndarray):
    ubar = np.mean(u, axis=1, keepdims=True)
    wbar = np.mean(w, axis=1, keepdims=True)
    up = u - ubar
    wp = w - wbar
    return up, wp


def reynolds_stress_means(u: np.ndarray, w: np.ndarray):
    up, wp = reynolds_stress_fields(u, w)
    return {
        "uu": float(np.mean(up**2)),
        "ww": float(np.mean(wp**2)),
        "uw": float(np.mean(up * wp)),
    }


def reynolds_stress_profiles(u: np.ndarray, w: np.ndarray):
    up, wp = reynolds_stress_fields(u, w)
    return {
        "uu": np.mean(up**2, axis=1),
        "ww": np.mean(wp**2, axis=1),
        "uw": np.mean(up * wp, axis=1),
    }


def compute_energy_spectrum(u: np.ndarray, w: np.ndarray, lx: float, ly: float):
    '''
    Isotropic 1D kinetic-energy spectrum E11(k1) from 2D velocity fields.
    '''
    ny, nx = u.shape
    uhat = np.fft.fft2(u)
    what = np.fft.fft2(w)

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    kmag = np.sqrt(KX**2 + KY**2)

    energy_2d = 0.5 * (np.abs(uhat)**2 + np.abs(what)**2) / (nx * ny) ** 2

    dk = min(2.0 * np.pi / lx, 2.0 * np.pi / ly)
    kbins = np.arange(0.0, kmag.max() + dk, dk)
    kcenters = 0.5 * (kbins[:-1] + kbins[1:])
    Ek = np.zeros_like(kcenters)

    for i in range(len(kcenters)):
        mask = (kmag >= kbins[i]) & (kmag < kbins[i + 1])
        Ek[i] = np.sum(energy_2d[mask])

    valid = Ek > 0.0
    return kcenters[valid], Ek[valid]


def spectrum_to_wavelength(k: np.ndarray, Ek: np.ndarray):
    good = k > 0.0
    lam = 2.0 * np.pi / k[good]
    E = Ek[good]
    order = np.argsort(lam)
    return lam[order], E[order]


def compute_metrics(
    omega: np.ndarray,
    b: np.ndarray,
    y: np.ndarray,
    KX: np.ndarray,
    KY: np.ndarray,
    K2: np.ndarray,
    lx: float,
    ly: float,
) -> Dict[str, object]:
    _, u, w = velocity_from_vorticity(omega, KX, KY, K2)
    rs = reynolds_stress_means(u, w)
    k_spec, E_spec = compute_energy_spectrum(u, w, lx, ly)
    return {
        "u": u,
        "w": w,
        "tke": total_tke(u, w),
        "pke": perturbation_ke(u, w),
        "enstrophy": enstrophy(omega),
        "w_rms": vertical_velocity_rms(w),
        "mixing_thickness": mixing_thickness(b, y),
        "rs_uu": rs["uu"],
        "rs_ww": rs["ww"],
        "rs_uw": rs["uw"],
        "k_spectrum": k_spec,
        "E_spectrum": E_spec,
    }

#Time stepping#
# %%
def rhs(
    omega: np.ndarray,
    b: np.ndarray,
    KX: np.ndarray,
    KY: np.ndarray,
    K2: np.ndarray,
    nu: float,
    kappa: float,
    mask: Optional[np.ndarray],
):
    omega_hat = np.fft.fft2(omega)
    b_hat = np.fft.fft2(b)
    psi_hat = solve_minus_laplacian(omega_hat, K2)

    domega_hat = (
        -jacobian_hat(psi_hat, omega_hat, KX, KY, mask)
        + deriv_x_hat(b_hat, KX)
        + nu * laplacian_hat(omega_hat, K2)
    )
    db_hat = (
        -jacobian_hat(psi_hat, b_hat, KX, KY, mask)
        + kappa * laplacian_hat(b_hat, K2)
    )

    return to_real(apply_mask(domega_hat, mask)), to_real(apply_mask(db_hat, mask))

#RK4 time stepping#
# %%
def rk4_step(
    omega: np.ndarray,
    b: np.ndarray,
    dt: float,
    KX: np.ndarray,
    KY: np.ndarray,
    K2: np.ndarray,
    nu: float,
    kappa: float,
    mask: Optional[np.ndarray],
):
    k1_om, k1_b = rhs(omega, b, KX, KY, K2, nu, kappa, mask)
    k2_om, k2_b = rhs(omega + 0.5 * dt * k1_om, b + 0.5 * dt * k1_b, KX, KY, K2, nu, kappa, mask)
    k3_om, k3_b = rhs(omega + 0.5 * dt * k2_om, b + 0.5 * dt * k2_b, KX, KY, K2, nu, kappa, mask)
    k4_om, k4_b = rhs(omega + dt * k3_om, b + dt * k3_b, KX, KY, K2, nu, kappa, mask)

    omega_new = omega + dt * (k1_om + 2 * k2_om + 2 * k3_om + k4_om) / 6.0
    b_new = b + dt * (k1_b + 2 * k2_b + 2 * k3_b + k4_b) / 6.0
    return omega_new, b_new

#Run + save#
# %%
def run_simulation(cfg: KHConfig) -> Dict[str, object]:
    outpath = Path(cfg.outdir)
    outpath.mkdir(parents=True, exist_ok=True)

    KX, KY, K2 = make_wavenumbers(cfg.nx, cfg.ny, cfg.lx, cfg.ly)
    mask = dealias_mask(cfg.nx, cfg.ny)

    X, Y, omega, b, U_base, b_base = initialize_fields(cfg)
    y = Y[:, 0]

    history = {
        name: []
        for name in [
            "times",
            "tke",
            "pke",
            "enstrophy",
            "w_rms",
            "mixing_thickness",
            "growth",
            "reynolds_uu",
            "reynolds_ww",
            "reynolds_uw",
        ]
    }
    snapshots = {name: [] for name in ["t", "omega", "b", "u", "w", "k_spectrum", "E_spectrum"]}

    mix0 = mixing_thickness(b, y)
    pke0 = None

    for n in range(cfg.nt + 1):
        t = n * cfg.dt
        metrics = compute_metrics(omega, b, y, KX, KY, K2, cfg.lx, cfg.ly)

        if pke0 is None:
            pke0 = max(metrics["pke"], 1e-20)

        history["times"].append(t)
        history["tke"].append(metrics["tke"])
        history["pke"].append(metrics["pke"])
        history["enstrophy"].append(metrics["enstrophy"])
        history["w_rms"].append(metrics["w_rms"])
        history["mixing_thickness"].append(metrics["mixing_thickness"])
        history["growth"].append(metrics["pke"] / pke0)
        history["reynolds_uu"].append(metrics["rs_uu"])
        history["reynolds_ww"].append(metrics["rs_ww"])
        history["reynolds_uw"].append(metrics["rs_uw"])

        if n % cfg.snapshot_every == 0 or n == cfg.nt:
            snapshots["t"].append(t)
            snapshots["omega"].append(omega.copy())
            snapshots["b"].append(b.copy())
            snapshots["u"].append(metrics["u"].copy())
            snapshots["w"].append(metrics["w"].copy())
            snapshots["k_spectrum"].append(metrics["k_spectrum"].copy())
            snapshots["E_spectrum"].append(metrics["E_spectrum"].copy())

            if cfg.verbose:
                print(
                    f"[{cfg.label}] step {n:6d}/{cfg.nt}  "
                    f"t={t:7.3f}  TKE={metrics['tke']:.4e}  "
                    f"PKE={metrics['pke']:.4e}  Wrms={metrics['w_rms']:.4e}  "
                    f"Ens={metrics['enstrophy']:.4e}  Mix={metrics['mixing_thickness']:.4e}  "
                    f"<u'w'>={metrics['rs_uw']:.4e}"
                )

        if n < cfg.nt:
            omega, b = rk4_step(omega, b, cfg.dt, KX, KY, K2, cfg.nu, cfg.kappa, mask)

    for key in history:
        history[key] = np.asarray(history[key], dtype=float)
    for key in ["t", "omega", "b", "u", "w", "k_spectrum", "E_spectrum"]:
        snapshots[key] = np.asarray(snapshots[key], dtype=object if "spectrum" in key else float)

    growth_max = float(np.max(history["growth"]))
    peak_tke = float(np.max(history["tke"]))
    peak_enstrophy = float(np.max(history["enstrophy"]))
    peak_wrms = float(np.max(history["w_rms"]))
    final_mixing = float(history["mixing_thickness"][-1])
    mix_ratio = final_mixing / max(mix0, 1e-20)
    transition_time = estimate_transition_time(history["times"], history["pke"])
    sigma, intercept, r2, fit_idx = fit_growth_rate(history["times"], history["pke"])
    regime = classify_regime(growth_max, peak_enstrophy, mix_ratio, transition_time) #Add code for making regime map

    results = {
        "X": X,
        "Y": Y,
        "times": history["times"],
        "tke": history["tke"],
        "pke": history["pke"],
        "enstrophy": history["enstrophy"],
        "w_rms": history["w_rms"],
        "mixing_thickness": history["mixing_thickness"],
        "growth": history["growth"],
        "reynolds_uu": history["reynolds_uu"],
        "reynolds_ww": history["reynolds_ww"],
        "reynolds_uw": history["reynolds_uw"],
        "mix0": float(mix0),
        "snapshots": snapshots,
        "params": asdict(cfg),
        "growth_fit": {
            "sigma": sigma,
            "intercept": intercept,
            "r2": r2,
            "fit_idx": fit_idx,
        },
        "summary": {
            "peak_tke": peak_tke,
            "peak_enstrophy": peak_enstrophy,
            "peak_wrms": peak_wrms,
            "final_mixing_thickness": final_mixing,
            "mix_ratio": mix_ratio,
            "transition_time": transition_time,
            "growth_max": growth_max,
            "regime": regime,
        },
    }

    fit = results["growth_fit"]
    summary = results["summary"]
    snap = results["snapshots"]

    np.savez(
        outpath / f"{cfg.label}_results.npz",
        times=results["times"],
        tke=results["tke"],
        pke=results["pke"],
        enstrophy=results["enstrophy"],
        w_rms=results["w_rms"],
        mixing_thickness=results["mixing_thickness"],
        growth=results["growth"],
        reynolds_uu=results["reynolds_uu"],
        reynolds_ww=results["reynolds_ww"],
        reynolds_uw=results["reynolds_uw"],
        mix0=results["mix0"],
        Ri=cfg.Ri,
        Re=cfg.Re,
        Pr=cfg.Pr,
        lx=cfg.lx,
        ly=cfg.ly,
        sigma=fit["sigma"],
        fit_r2=fit["r2"],
        peak_tke=summary["peak_tke"],
        peak_enstrophy=summary["peak_enstrophy"],
        peak_wrms=summary["peak_wrms"],
        final_mixing_thickness=summary["final_mixing_thickness"],
        mix_ratio=summary["mix_ratio"],
        transition_time=-1.0 if np.isnan(summary["transition_time"]) else summary["transition_time"],
        growth_max=summary["growth_max"],
        X=results["X"],
        Y=results["Y"],
        snapshot_times=snap["t"],
        snapshot_omega=snap["omega"],
        snapshot_b=snap["b"],
        snapshot_u=snap["u"],
        snapshot_w=snap["w"],
        snapshot_k=snap["k_spectrum"],
        snapshot_Ek=snap["E_spectrum"],
    )

    return results

#Plotting helpers#
# %%
def plot_time_series(results: Dict[str, object]) -> plt.Figure:
    t = results["times"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)

    panels = [
        ("tke", "Turbulent kinetic energy"),
        ("pke", "Perturbation kinetic energy"),
        ("enstrophy", "Enstrophy"),
        ("w_rms", "Vertical velocity RMS"),
        ("mixing_thickness", "Mixing thickness"),
        ("growth", "PKE / PKE(0)"),
        ("reynolds_uu", r"$\langle u'u' \rangle$"),
        ("reynolds_ww", r"$\langle w'w' \rangle$"),
        ("reynolds_uw", r"$\langle u'w' \rangle$"),
    ]

    # Add a main title for the figure if Ri and Re are available in results
    main_title = "Time Series"
    if 'Ri' in results and 'Re' in results:
        main_title = f"Time Series (Ri={results['Ri']}, Re={results['Re']})"
    # Increased y from 1.02 to 1.05 for more space
    fig.suptitle(main_title, fontsize=16, y=1.05)

    for ax, (key, ylabel) in zip(axes.flat, panels):
        if key == "growth":
            ax.semilogy(t, np.maximum(results[key], 1e-30), linewidth=2)
        else:
            ax.plot(t, results[key], linewidth=2)
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)

    # Do not call plt.show() here, return the figure instead.
    return fig

#User settings#
# %%
cfg = KHConfig(
    nx=192,
    ny=192,
    lx=14.0,
    ly=14.0,
    tmax=100.0,
    dt=2e-3,
    Ri=0.15,
    Re=1000.0,
    Pr=1.0,
    perturbation_amplitude=1e-3,
    perturbation_mode=2,
    U0=1.0,
    shear_thickness=0.5,
    density_thickness=0.5,
    perturbation_phase=0.2,
    noise_amplitude=0.0,
    seed=0,
    snapshot_every=250,
    outdir="kh_butterfly_raw",
    label=f"butterfly",
    verbose=True,
)
# %%
cfg
# %%
#Check domain setup
X, Y, omega_initial, b_initial, U_base, b_base = initialize_fields(cfg)

# Compute 2D FFT of initial omega and b
omega_initial_hat = np.fft.fft2(omega_initial)
b_initial_hat = np.fft.fft2(b_initial)

# Shift the zero-frequency component to the center for visualization
omega_initial_hat_shifted = np.fft.fftshift(omega_initial_hat)
b_initial_hat_shifted = np.fft.fftshift(b_initial_hat)

fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

# Plot initial omega in physical space
im0 = axes[0, 0].imshow(omega_initial, origin="lower", aspect="auto", cmap="RdBu_r")
axes[0, 0].set_title("Initial Vorticity (Physical Space)")
plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)

# Plot initial b in physical space
im1 = axes[0, 1].imshow(b_initial, origin="lower", aspect="auto", cmap="viridis")
axes[0, 1].set_title("Initial Buoyancy (Physical Space)")
plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)

# Plot magnitude of omega_hat (FFT mesh) in spectral space
im2 = axes[1, 0].imshow(np.log(np.abs(omega_initial_hat_shifted) + 1e-10), origin="lower", aspect="auto", cmap="magma")
axes[1, 0].set_title("Log Magnitude of Initial Vorticity FFT (Spectral Space)")
plt.colorbar(im2, ax=axes[1, 0], shrink=0.8)

# Plot magnitude of b_hat (FFT mesh) in spectral space
im3 = axes[1, 1].imshow(np.log(np.abs(b_initial_hat_shifted) + 1e-10), origin="lower", aspect="auto", cmap="magma")
axes[1, 1].set_title("Log Magnitude of Initial Buoyancy FFT (Spectral Space)")
plt.colorbar(im3, ax=axes[1, 1], shrink=0.8)

plt.show()

#Run one simulation#
# %%
results = run_simulation(cfg)
results["summary"]
print(f"Saved results to: {Path(cfg.outdir) / f'{cfg.label}_results.npz'}")

"""## RESULTS FOR ONE SIMULATION

### Detailed 2D Snapshots with Velocity Vectors

These plots show a selected snapshot of the vorticity and buoyancy fields, with velocity vectors overlaid to illustrate the fluid motion. The vectors provide insight into how the flow is organized within the structures revealed by the scalar fields.
"""

outpath = Path(cfg.outdir)

snapshot_index_detailed =-1 #27, 29, -1, 32, 100, 120(last snapshot)

snapshot_omega = results["snapshots"]["omega"][snapshot_index_detailed]
snapshot_b = results["snapshots"]["b"][snapshot_index_detailed]
snapshot_u = results["snapshots"]["u"][snapshot_index_detailed]
snapshot_w = results["snapshots"]["w"][snapshot_index_detailed]

X_coords = results["X"]
Y_coords = results["Y"]

# Create a quiver plot for velocity vectors. Subsample to avoid cluttered plot.
skip = 8 # Adjust this value to change vector density

fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
# %%
# Plot Vorticity with Quiver
im0 = axes[0].imshow(snapshot_omega, origin="lower", aspect="auto", cmap="RdBu_r",
                     extent=[X_coords.min(), X_coords.max(), Y_coords.min(), Y_coords.max()])
axes[0].quiver(X_coords[::skip, ::skip], Y_coords[::skip, ::skip],
               snapshot_u[::skip, ::skip], snapshot_w[::skip, ::skip], color='k', scale=20, width=0.003)
axes[0].set_title(f"Vorticity with Velocity Vectors (Ri={results['params']['Ri']}, Re={results['params']['Re']}, Snapshot {snapshot_index_detailed})")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
plt.colorbar(im0, ax=axes[0], shrink=0.8)
# %%
# Plot Buoyancy with Quiver
im1 = axes[1].imshow(snapshot_b, origin="lower", aspect="auto", cmap="viridis",
                     extent=[X_coords.min(), X_coords.max(), Y_coords.min(), Y_coords.max()])
axes[1].quiver(X_coords[::skip, ::skip], Y_coords[::skip, ::skip],
               snapshot_u[::skip, ::skip], snapshot_w[::skip, ::skip], color='w', scale=20, width=0.003)
axes[1].set_title(f"Buoyancy with Velocity Vectors (Ri={results['params']['Ri']}, Re={results['params']['Re']}, Snapshot {snapshot_index_detailed})")
axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")
plt.colorbar(im1, ax=axes[1], shrink=0.8)
filename = "simulationwvectors_41.png"
plt.savefig(outpath / filename, dpi=180)
plt.show()

# plot time series for a single simulation result.
# %%
print(f"Plotting time series for Ri={results['params']['Ri']}, Re={results['params']['Re']}")
# Add the Ri and Re to the results dictionary for clearer titles in the plot_time_series function
results['Ri'] = results['params']['Ri']
results['Re'] = results['params']['Re']

# Call plot_time_series and get the figure object
fig = plot_time_series(results)

filename = "baseline_plots.png"
# Save the figure before showing it
plt.savefig(outpath / filename, dpi=300)
plt.show()

snapshot_index = 120 #27, 32, 60, 120
y = results["Y"][:, 0]
u_snap = results["snapshots"]["u"][snapshot_index]
w_snap = results["snapshots"]["w"][snapshot_index]
profiles = reynolds_stress_profiles(u_snap, w_snap)

plt.figure(figsize=(7, 5))
plt.plot(profiles["uu"], y, label=r"$\overline{u'u'}(y)$", linewidth=2)
plt.plot(profiles["ww"], y, label=r"$\overline{w'w'}(y)$", linewidth=2)
plt.plot(profiles["uw"], y, label=r"$\overline{u'w'}(y)$", linewidth=2)
plt.xlabel("Reynolds stress")
plt.ylabel("y")
plt.title("Reynolds-stress profiles")
plt.legend()
plt.tight_layout()
plt.show()

filename = "RE_baseline_plot120_.png"
# Save the figure before showing it
plt.savefig(outpath / filename, dpi=300)
plt.show()
# %%
snapshot_index = 60
#changes at 10, 32, 38, 39, -1
k = np.asarray(results["snapshots"]["k_spectrum"][snapshot_index], dtype=float)
Ek = np.asarray(results["snapshots"] ["E_spectrum"][snapshot_index], dtype=float)

plt.figure(figsize=(7, 5))
plt.loglog(k, Ek, linewidth=2, label="E(k)")

if len(k) > 5:
    iref = len(k) // 2
    ref = Ek[iref] * (k / k[iref]) ** (-5.0 / 3.0)
    plt.loglog(k, ref, "--", linewidth=1.5, label=r"$k^{-5/3}$")

if len(k) > 5:
    iref = len(k) // 2
    ref = Ek[iref] * (k / k[iref]) ** (-3.0)
    plt.loglog(k, ref, "--", linewidth=1.5, label=r"$k^{-3}$")

if len(k) > 5:
    iref = len(k) // 2
    ref = Ek[iref] * (k / k[iref]) ** (-2.0)
    plt.loglog(k, ref, "--", linewidth=1.5, label=r"$k^{-2}$")

plt.xlabel(r"Wavenumber $k$")
plt.ylabel(r"$E(k)$")
plt.title("Kinetic-energy spectrum")
plt.legend()
plt.tight_layout()
plt.show()

filename = "EK_baseline_plot60.png"
# Save the figure before showing it
plt.savefig(outpath / filename, dpi=300)
plt.show()

# Ensure Ri and Re are directly available in the results dictionary for plotting function
if 'Ri' not in results:
    results['Ri'] = results['params']['Ri']
if 'Re' not in results:
    results['Re'] = results['params']['Re']

# Choose a few snapshot indices to visualize the evolution of the energy spectrum
snapshot_indices_to_plot = [32, 120] # These indices correspond to t=0, 10, 20, 30, 40

print(f"Plotting kinetic energy spectra at various times for Ri={results['Ri']}, Re={results['Re']}:")

plt.figure(figsize=(10, 7))
reference_drawn = False

for idx in snapshot_indices_to_plot:
    if idx < len(results["snapshots"]["k_spectrum"]):
        time_at_snapshot = results["snapshots"]["t"][idx]
        print(f"  Snapshot index {idx} (t={time_at_snapshot:.2f})")

        k = np.asarray(results["snapshots"]["k_spectrum"][idx], dtype=float)
        Ek = np.asarray(results["snapshots"]["E_spectrum"][idx], dtype=float)

        plt.loglog(k, Ek, linewidth=2, label=f"t={time_at_snapshot:.2f}")

        # Plot k^-5/3 reference line only once
        if not reference_drawn and len(k) > 5:
            iref = len(k) // 2
            ref_k53 = Ek[iref] * (k / k[iref]) ** (-5.0 / 3.0)
            plt.loglog(k, ref_k53, "k--", linewidth=1.5, label=r"$k^{-5/3}$ reference")
            reference_drawn = True
    else:
        print(f"Warning: Snapshot index {idx} is out of bounds. Max index is {len(results['snapshots']['k_spectrum']) - 1}. Skipping.")

plt.xlabel(r"Wavenumber $k$")
plt.ylabel(r"$E(k)$")
plt.title(f"Kinetic-energy spectrum (Ri={results['Ri']}, Re={results['Re']})")
plt.legend()
plt.grid(True, which="both", ls="-")
plt.tight_layout()
filename = "baseline_EkplotCompare2.png"
# Save the figure before showing it
plt.savefig(outpath / filename, dpi=300)
plt.show()

#Full Parameter Sweep: Varying Ri and Re#
# %%
#takes about >3.5 hours to run for 3x3
Ri_list = [0.05, 0.10, 0.20]
Re_list = [500.0, 1500.0, 2000.0]

#Ri_list = [0.05, 0.1, 0.15, 0.2]
#Re_list = [500.0, 1000.0, 1500.0, 2000.0]
# Initialize an empty grid to store the chosen metric (e.g., growth_max)
grid = np.zeros((len(Ri_list), len(Re_list)))

# Assuming `cfg` is the base configuration from earlier
cfg = KHConfig(
    nx=192,
    ny=192,
    lx=14.0,
    ly=14.0,
    tmax=60.0,
    dt=2e-3,
    Ri=0.15,
    Re=1000.0,
    Pr=1.0,
    perturbation_amplitude=1e-3,
    perturbation_mode=2,
    U0=1.0,
    shear_thickness=0.5,
    density_thickness=0.5,
    perturbation_phase=0.0,
    noise_amplitude=0.0,
    seed=0,
    snapshot_every=250,
    outdir="kh_sweeps_raw",
    label="baseline",
    verbose=True,
)

base_cfg = cfg # Use the existing configuration as a template

outpath = Path(base_cfg.outdir)
outpath.mkdir(parents=True, exist_ok=True)

print("Running parameter sweep...")
for i, current_Ri in enumerate(Ri_list):
    for j, current_Re in enumerate(Re_list):
        print(f"  Running Ri={current_Ri}, Re={current_Re}...")
        # Create a new config for each run
        run_cfg = KHConfig(
            nx=base_cfg.nx,
            ny=base_cfg.ny,
            lx=base_cfg.lx,
            ly=base_cfg.ly,
            tmax=base_cfg.tmax,
            dt=base_cfg.dt,
            Ri=current_Ri,
            Re=current_Re,
            Pr=base_cfg.Pr,
            U0=base_cfg.U0,
            shear_thickness=base_cfg.shear_thickness,
            density_thickness=base_cfg.density_thickness,
            perturbation_amplitude=base_cfg.perturbation_amplitude,
            perturbation_mode=base_cfg.perturbation_mode,
            perturbation_phase=base_cfg.perturbation_phase,
            noise_amplitude=base_cfg.noise_amplitude,
            seed=base_cfg.seed,
            snapshot_every=base_cfg.snapshot_every,
            outdir=base_cfg.outdir,
            label=f"Ri{current_Ri}_Re{current_Re}", # Unique label for each run
            verbose=False # Set to True to see individual run output
        )

        # Run the simulation
        current_results = run_simulation(run_cfg)

        # Store the desired metric (e.g., growth_max) in the grid
        grid[i, j] = current_results["summary"]["final_mixing_thickness"]

print("Parameter sweep complete.")
# %%
all_sweep_results = {}
outdir_path = Path(cfg.outdir)

print("Loading parameter sweep results...")
for current_Ri in Ri_list:
    for current_Re in Re_list:
        label = f"Ri{current_Ri}_Re{current_Re}"
        npz_path = outdir_path / f"{label}_results.npz"

        if npz_path.exists():
            data_sweep = dict(np.load(npz_path, allow_pickle=True))
            all_sweep_results[(current_Ri, current_Re)] = data_sweep
            print(f"  Loaded results for Ri={current_Ri}, Re={current_Re}")
        else:
            print(f"Warning: {npz_path} not found. Skipping loading for this combination.")

print(f"Loaded {len(all_sweep_results)} parameter sweep results.")

#RESULTS FOR FULL SWEEP
#Will need to update results or all_sweeps_results (pick which Re,Ri sweep)

# %%
snapshot_index_detailed = 32 #27, 29, -1, 32

# Select one specific result from the sweep to plot
# You can change (Ri, Re) here to view different simulation results
selected_ri = Ri_list[0] # Example: choose the first Ri from Ri_list
selected_re = Re_list[0] # Example: choose the first Re from Re_list
selected_results = all_sweep_results[(selected_ri, selected_re)]

# Retrieve data for the chosen snapshot from the selected results
snapshot_omega = all_sweep_results["snapshot_omega"][snapshot_index_detailed]
snapshot_b = all_sweep_results["snapshot_b"][snapshot_index_detailed]
snapshot_u = all_sweep_results["snapshot_u"][snapshot_index_detailed]
snapshot_w = all_sweep_results["snapshot_w"][snapshot_index_detailed]

X_coords = all_sweep_results["X"]
Y_coords = all_sweep_results["Y"]

# Create a quiver plot for velocity vectors. Subsample to avoid cluttered plot.
skip = 8 # Adjust this value to change vector density

fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

# Plot Vorticity with Quiver
im0 = axes[0].imshow(snapshot_omega, origin="lower", aspect="auto", cmap="RdBu_r",
                     extent=[X_coords.min(), X_coords.max(), Y_coords.min(), Y_coords.max()])
axes[0].quiver(X_coords[::skip, ::skip], Y_coords[::skip, ::skip],
               snapshot_u[::skip, ::skip], snapshot_w[::skip, ::skip], color='k', scale=20, width=0.003)
axes[0].set_title(f"Vorticity with Velocity Vectors (Ri={selected_ri}, Re={selected_re}, Snapshot {snapshot_index_detailed})")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

# Plot Buoyancy with Quiver
im1 = axes[1].imshow(snapshot_b, origin="lower", aspect="auto", cmap="viridis",
                     extent=[X_coords.min(), X_coords.max(), Y_coords.min(), Y_coords.max()])
axes[1].quiver(X_coords[::skip, ::skip], Y_coords[::skip, ::skip],
               snapshot_u[::skip, ::skip], snapshot_w[::skip, ::skip], color='w', scale=20, width=0.003)
axes[1].set_title(f"Buoyancy with Velocity Vectors (Ri={selected_ri}, Re={selected_re}, Snapshot {snapshot_index_detailed})")
axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")
plt.colorbar(im1, ax=axes[1], shrink=0.8)
#filename = "simulationwvectors.png"
#plt.savefig(outpath / filename, dpi=180)
plt.show()

for (ri, re), results_dict in all_sweep_results.items():
    print(f"Plotting time series for Ri={ri}, Re={re}")
    # Add the Ri and Re to the results_dict for clearer titles in the plot_time_series function
    results_dict['Ri'] = ri
    results_dict['Re'] = re
    plot_time_series(results_dict)

#Comparison of All Time Series Metrics Across Parameter Sweeps
#These plots visualize the time evolution of various metrics for all combinations of Richardson number (Ri) and Reynolds number (Re) explored in the parameter sweep.

# %%
import matplotlib.pyplot as plt
import numpy as np

# Define the metrics to plot and their display names
metrics_to_plot = [
    ("tke", "Total Kinetic Energy (TKE)"),
    ("pke", "Perturbation Kinetic Energy (PKE)"),
    ("enstrophy", "Enstrophy"),
    ("w_rms", "Vertical Velocity RMS"),
    ("mixing_thickness", "Mixing Layer Thickness"),
    ("growth", "PKE / PKE(0) (Growth Factor)"),
    ("reynolds_uu", r"Reynolds Stress $\langle u'u' \rangle$"),
    ("reynolds_ww", r"Reynolds Stress $\langle w'w' \rangle$"),
    ("reynolds_uw", r"Reynolds Stress $\langle u'w' \rangle$"),
]

# Iterate through each metric and create a plot
for metric_key, metric_label in metrics_to_plot:
    plt.figure(figsize=(12, 8))

    for (ri, re), results_dict in all_sweep_results.items():
        times = results_dict["times"]
        # Ensure Ri and Re are available in the results_dict for plotting titles
        results_dict['Ri'] = ri
        results_dict['Re'] = re

        # Safely get the metric data, handling potential missing keys or non-numeric types
        if metric_key in results_dict and isinstance(results_dict[metric_key], np.ndarray):
            data = results_dict[metric_key]

            # Apply semilogy for 'growth' or when data spans many orders of magnitude
            if metric_key == "growth" or (data.max() / data.min() > 1e3 and data.min() > 0):
                plt.semilogy(times, np.maximum(data, 1e-30), label=f"Ri={ri}, Re={re}") # Use maximum to avoid log(0)
            else:
                plt.plot(times, data, label=f"Ri={ri}, Re={re}")
        else:
            print(f"Warning: Metric '{metric_key}' not found or not suitable for plotting in Ri={ri}, Re={re} results.")

    plt.xlabel("Time")
    plt.ylabel(metric_label)
    plt.title(f"Time Evolution of {metric_label} Across All Sweeps")
    plt.legend(title="Simulation Parameters", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-")
    plt.tight_layout()
    filename = f"{metric_key}_time_evolution_allsweeps.png"
    plt.savefig(outpath / filename, dpi=180)
    plt.show()

#2D Plots of Vorticity and Buoyancy

#These plots show the initial 2D fields of vorticity ($\omega$) and buoyancy ($b$).

#Show one saved snapshot

# %%
snapshot_index = 36
#27-36 shows well for billows
#37-44 shows well for turbulent dissipation

fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

im0 = axes[0].imshow(all_sweep_results["snapshot_omega"][snapshot_index], origin="lower", aspect="auto")
axes[0].set_title("Vorticity")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(all_sweep_results["snapshot_b"][snapshot_index], origin="lower", aspect="auto")
axes[1].set_title("Buoyancy")
plt.colorbar(im1, ax=axes[1], shrink=0.8)

im2 = axes[2].imshow(all_sweep_results["snapshot_w"][snapshot_index], origin="lower", aspect="auto")
axes[2].set_title("Vertical velocity")
plt.colorbar(im2, ax=axes[2], shrink=0.8)

plt.show()

#Load an existing `.npz` result file#
# %%
selected_ri = Ri_list[1] #[0] calls which one
selected_re = Re_list[1]
cfg.label = f"Ri{selected_ri}_Re{selected_re}"

npz_path = Path(cfg.outdir) / f"{cfg.label}_results.npz"
data = dict(np.load(npz_path, allow_pickle=True))
#sorted(data.keys())
print(f"Loaded data from {npz_path}", cfg.label)

#Plot Reynolds-stress summary from saved data#
# %%
def plot_reynolds_stress_summary(data: dict, snapshot_index: int) -> None:
    y = data["Y"][:, 0]
    u_snap = data["snapshot_u"][snapshot_index]
    w_snap = data["snapshot_w"][snapshot_index]
    profiles = reynolds_stress_profiles(u_snap, w_snap)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(profiles["uu"], y, label=r"$\overline{u'u'}(y)$", linewidth=2)
    ax.plot(profiles["ww"], y, label=r"$\overline{w'w'}(y)$", linewidth=2)
    ax.plot(profiles["uw"], y, label=r"$\overline{u'w'}(y)$", linewidth=2)
    ax.set_xlabel("Reynolds stress")
    ax.set_ylabel("y")

    title_parts = ["Reynolds-stress profiles"]
    # Check if Ri and Re are available in the data dictionary
    # Note: data in this context is loaded from an .npz file, which stores Ri and Re directly.
    if 'Ri' in data and 'Re' in data:
        title_parts.append(f"(Ri={data['Ri']}, Re={data['Re']})")
    title_parts.append(f"Snapshot {snapshot_index}")
    ax.set_title(" ".join(title_parts))

    ax.legend()
    plt.tight_layout()
    plt.show()

plot_reynolds_stress_summary(data, snapshot_index=27)

#Plot energy-spectrum summary from saved data#
# %%
def plot_energy_spectrum_summary(data: dict, snapshot_index: int, show_k53: bool = False, show_k3: bool = False) -> None:
    k = np.asarray(data["snapshot_k"][snapshot_index], dtype=float)
    Ek = np.asarray(data["snapshot_Ek"][snapshot_index], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.loglog(k, Ek, linewidth=2, label="E(k)")

    if show_k53 and len(k) > 5:
        iref = len(k) // 2
        ref_k53 = Ek[iref] * (k / k[iref]) ** (-5.0 / 3.0)
        ax.loglog(k, ref_k53, "--", linewidth=1.5, label=r"$k^{-5/3}$")

    if show_k3 and len(k) > 5:
        iref = len(k) // 2
        ref_k3 = Ek[iref] * (k / k[iref]) ** (-3.0)
        ax.loglog(k, ref_k3, "--", linewidth=1.5, label=r"$k^{-3}$")

    ax.set_xlabel(r"Wavenumber $k$")
    ax.set_ylabel(r"$E(k)$")

    title_parts = ["Kinetic-energy spectrum"]
    if 'Ri' in data and 'Re' in data:
        title_parts.append(f"(Ri={data['Ri']}, Re={data['Re']})")
    title_parts.append(f"Snapshot {snapshot_index}")
    ax.set_title(" ".join(title_parts))

    ax.legend()
    plt.tight_layout()
    plt.show()

plot_energy_spectrum_summary(data, snapshot_index=20, show_k53=True)
# %%
snapshot_index = 60
#changes at 10, 32, 38, 39, -1
k = np.asarray(data["snapshot_k"][snapshot_index], dtype=float)
Ek = np.asarray(data["snapshot_Ek"][snapshot_index], dtype=float)

plt.figure(figsize=(7, 5))
plt.loglog(k, Ek, linewidth=2, label="E(k)")

if len(k) > 5:
    iref = len(k) // 2
    ref = Ek[iref] * (k / k[iref]) ** (-5.0 / 3.0)
    plt.loglog(k, ref, "--", linewidth=1.5, label=r"$k^{-5/3}$")

if len(k) > 5:
    iref = len(k) // 2
    ref = Ek[iref] * (k / k[iref]) ** (-3.0)
    plt.loglog(k, ref, "--", linewidth=1.5, label=r"$k^{-3}$")

if len(k) > 5:
    iref = len(k) // 2
    ref = Ek[iref] * (k / k[iref]) ** (-2.0)
    plt.loglog(k, ref, "--", linewidth=1.5, label=r"$k^{-2}$")

plt.xlabel(r"Wavenumber $k$")
plt.ylabel(r"$E(k)$")
plt.title("0.10 Ri, Re 1500 Kinetic-energy spectrum")
plt.legend()
plt.tight_layout()
plt.savefig(outpath / "Ek_allscales.png", dpi=180)
plt.show()

# Ensure Ri and Re are directly available in the results dictionary for plotting function
if 'Ri' not in results:
    results['Ri'] = results['params']['Ri']
if 'Re' not in results:
    results['Re'] = results['params']['Re']

# Choose a few snapshot indices to visualize the evolution of the energy spectrum
snapshot_indices_to_plot = [0, 20, 40, 60, 80] # These indices correspond to t=0, 10, 20, 30, 40

print(f"Plotting kinetic energy spectra at various times for Ri={results['Ri']}, Re={results['Re']}:")

plt.figure(figsize=(10, 7))
reference_drawn = False

for idx in snapshot_indices_to_plot:
    if idx < len(results["snapshots"]["k_spectrum"]):
        time_at_snapshot = results["snapshots"]["t"][idx]
        print(f"  Snapshot index {idx} (t={time_at_snapshot:.2f})")

        k = np.asarray(results["snapshots"]["k_spectrum"][idx], dtype=float)
        Ek = np.asarray(results["snapshots"]["E_spectrum"][idx], dtype=float)

        plt.loglog(k, Ek, linewidth=2, label=f"t={time_at_snapshot:.2f}")

        # Plot k^-5/3 reference line only once
        if not reference_drawn and len(k) > 5:
            iref = len(k) // 2
            ref_k53 = Ek[iref] * (k / k[iref]) ** (-5.0 / 3.0)
            plt.loglog(k, ref_k53, "k--", linewidth=1.5, label=r"$k^{-5/3}$ reference")
            reference_drawn = True
    else:
        print(f"Warning: Snapshot index {idx} is out of bounds. Max index is {len(results['snapshots']['k_spectrum']) - 1}. Skipping.")

plt.xlabel(r"Wavenumber $k$")
plt.ylabel(r"$E(k)$")
plt.title(f"Kinetic-energy spectrum (Ri={results['Ri']}, Re={results['Re']})")
plt.legend()
plt.grid(True, which="both", ls="-")
plt.tight_layout()
plt.show()

#Creating an animated GIF from snapshots

#To visualize the evolution of the fields over time, I created an animated GIF from the saved snapshots. This involves iterating through the snapshots, plotting each one, and then compiling these individual plots into a GIF.

# %%
# Install imageio
!pip install imageio

import imageio.v2 as imageio
import os
import shutil

def create_snapshot_gif(
    results_dict: dict,
    field_name: str,
    output_filename: str,
    fps: int = 10,
    cmap: str = 'RdBu_r',
    fig_size: tuple = (7, 6),
    title: Optional[str] = None
) -> None:
    """
    Creates an animated GIF from a series of 2D snapshots.

    Args:
        results_dict: A dictionary containing simulation results for a single run,
                      including 'X', 'Y', and snapshot data (e.g., 'snapshot_omega').
        field_name: The name of the field to animate (e.g., 'omega', 'b', 'u', 'w').
        output_filename: The name of the output GIF file (e.g., 'vorticity_animation.gif').
        fps: Frames per second for the GIF.
        cmap: Colormap to use for the imshow plot.
        fig_size: Tuple for the figure size (width, height).
        title: Optional title for each frame. If None, a default title is generated.
    """
    snapshot_data = results_dict['snapshots'][field_name]
    X_coords = results_dict['X']
    Y_coords = results_dict['Y']
    snapshot_times = results_dict['snapshots']['t']

    # Create a temporary directory for frames
    temp_dir = 'temp_frames_for_gif'
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Saving frames to {temp_dir}/")

    filenames = []
    for i, (snap, t) in enumerate(zip(snapshot_data, snapshot_times)):
        fig, ax = plt.subplots(figsize=fig_size)
        im = ax.imshow(snap, origin="lower", aspect="auto", cmap=cmap,
                       extent=[X_coords.min(), X_coords.max(), Y_coords.min(), Y_coords.max()])

        current_title = title if title else f"{field_name.capitalize()} at t={t:.2f}"
        ax.set_title(f"{current_title} (Ri={results_dict['params']['Ri']:.2f}, Re={results_dict['params']['Re']:.0f})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax, label=field_name.capitalize())
        plt.tight_layout()

        frame_filename = os.path.join(temp_dir, f'frame_{i:04d}.png')
        plt.savefig(frame_filename)
        plt.close(fig)
        filenames.append(frame_filename)

    print(f"Creating GIF: {output_filename}")
    with imageio.get_writer(output_filename, mode='I', fps=fps) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f"GIF saved as {output_filename}")

    # Clean up temporary frames directory
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary directory {temp_dir}/")

#Creates a GIF of the vorticity field for a specific simulation run. You can choose any `Ri` and `Re` combination from your `all_sweep_results`.
# %%
# Select a specific simulation run from sweep results
# For example, let's pick the first one from your defined Ri_list and Re_list
selected_ri_for_gif = Ri_list[2]
selected_re_for_gif = Re_list[0]

# Retrieve the results dictionary for this specific run
simulation_results_for_gif = all_sweep_results[(selected_ri_for_gif, selected_re_for_gif)]

# Add Ri and Re to the results_dict if not already present, for title generation in the function
simulation_results_for_gif['Ri'] = selected_ri_for_gif
simulation_results_for_gif['Re'] = selected_re_for_gif

# Define the output filename and field to animate
output_gif_name = f"vorticity_animation_Ri{selected_ri_for_gif}_Re{selected_re_for_gif}.gif"
field_to_animate = 'b' # Can be 'omega', 'b', 'u', or 'w'

# Call the function to create the GIF
create_snapshot_gif(
    simulation_results_for_gif,
    field_to_animate,
    output_gif_name,
    fps=10, # Adjust frames per second as desired
    cmap='RdBu_r' if field_to_animate == 'b' else 'viridis',
    title=f"Evolution of {field_to_animate.capitalize()}"
)

#Statistics and Review of Results#
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure outpath is defined, assuming cfg.outdir holds the correct base directory
outpath = Path(cfg.outdir)

# Re-load the data dictionary from all_sweep_results to ensure it's a dictionary
# Use the selected_ri and selected_re that were used for the GIF creation
# to maintain consistency with recent operations.
data = all_sweep_results[(selected_ri_for_gif, selected_re_for_gif)]

# Determine the canonical length from a known time series (e.g., 'times')
# This length will be used for all 1D arrays and for broadcasting scalars.
common_length = -1
if 'times' in data and isinstance(data['times'], np.ndarray) and data['times'].ndim == 1:
    common_length = len(data['times'])
else:
    # If 'times' is not suitable, try to find *any* 1D numeric array to establish common_length
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.ndim == 1 and np.issubdtype(value.dtype, np.number):
            common_length = len(value)
            break

if common_length == -1:
    raise ValueError("No suitable 1-dimensional numeric data found in 'data' to determine a common length for correlation.")

data_for_corr = {}

for key, value in data.items():
    # Check for 1D numeric arrays that match the common_length
    if isinstance(value, np.ndarray) and value.ndim == 1 and np.issubdtype(value.dtype, np.number):
        if len(value) == common_length:
            data_for_corr[key] = value
    # Check for 0D numeric arrays (scalars) or Python int/float, and broadcast them
    elif (isinstance(value, np.ndarray) and value.ndim == 0 and np.issubdtype(value.dtype, np.number)) or \
         (not isinstance(value, np.ndarray) and isinstance(value, (int, float))):
        data_for_corr[key] = np.full(common_length, value.item() if isinstance(value, np.ndarray) else value)

# Filter out constant columns (those with zero standard deviation)
if data_for_corr:
    temp_df = pd.DataFrame(data_for_corr)
    # Identify columns with standard deviation close to zero or explicit summary statistics to exclude
    constant_columns = [
        col for col in temp_df.columns if temp_df[col].std() < 1e-9 or col == 'growth_max'
    ]
    for col in constant_columns:
        del data_for_corr[col]

if not data_for_corr:
    raise ValueError("No suitable 1-dimensional numeric data found in 'data' for correlation matrix after filtering constant columns.")

corr = pd.DataFrame(data_for_corr).corr().round(2)
fig, ax = plt.subplots(figsize=(13, 10))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.index)))
ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
ax.set_yticklabels(corr.index, fontsize=7)
ax.set_title(f"Correlation Matrix — {len(corr.columns)} features", fontsize=11)
plt.tight_layout()
plt.savefig(outpath / "correlation_matrix.png", dpi=180)
plt.show()

pd.DataFrame(data_for_corr).describe().round(2)

#Comparison of Reynolds Stress Profiles Across Parameter Sweeps

#These plots visualize the vertical profiles of the three Reynolds stress components ($\langle u'u' \rangle$, $\langle w'w' \rangle$, and $\langle u'w' \rangle$) for all combinations of Richardson number (Ri) and Reynolds number (Re) explored in the parameter sweep.

# %%
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define the Reynolds stress components to plot and their display names
reynolds_stress_components = [
    ("reynolds_uu", r"Reynolds Stress $\langle u'u' \rangle$"),
    ("reynolds_ww", r"Reynolds Stress $\langle w'w' \rangle$"),
    ("reynolds_uw", r"Reynolds Stress $\langle u'w' \rangle$"),
]

# Choose a consistent snapshot index for all plots
# You might want to adjust this based on when significant structures develop
chosen_snapshot_index = 32 # Example: a mid-time snapshot, adjust as needed

# Ensure outpath is defined, assuming cfg.outdir holds the correct base directory
outpath = Path(cfg.outdir)

# Get the common y-coordinates from one of the results (they should all be the same)
# Assuming all_sweep_results is not empty
if all_sweep_results:
    _, first_results_dict = next(iter(all_sweep_results.items()))
    y = first_results_dict["Y"][:, 0]
else:
    raise ValueError("all_sweep_results is empty. Cannot determine y-coordinates.")

# Iterate through each Reynolds stress component and create a plot
for rs_key, rs_label in reynolds_stress_components:
    plt.figure(figsize=(8, 7))

    for (ri, re), results_dict in all_sweep_results.items():
        # Retrieve snapshot data for u and w
        # Ensure we handle potential data loading issues if snapshots are missing
        if "snapshot_u" in results_dict and "snapshot_w" in results_dict:
            u_snap = np.asarray(results_dict["snapshot_u"][chosen_snapshot_index], dtype=float)
            w_snap = np.asarray(results_dict["snapshot_w"][chosen_snapshot_index], dtype=float)

            # Compute Reynolds stress profiles for this snapshot
            profiles = reynolds_stress_profiles(u_snap, w_snap)

            # Plot the specific Reynolds stress component
            # The key mapping needs to be consistent: 'reynolds_uu' -> 'uu'
            plot_key = rs_key.replace("reynolds_", "")
            if plot_key in profiles:
                plt.plot(profiles[plot_key], y, label=f"Ri={ri}, Re={re}", linewidth=1.5)
            else:
                print(f"Warning: Reynolds stress component '{plot_key}' not found in profiles for Ri={ri}, Re={re}.")
        else:
            print(f"Warning: Snapshots for u or w not found in results for Ri={ri}, Re={re}. Skipping.")

    plt.xlabel(rs_label)
    plt.ylabel("y")
    plt.title(f"{rs_label} Profiles at t={first_results_dict['snapshot_times'][chosen_snapshot_index]:.1f} Across All Sweeps")
    plt.legend(title="Simulation Parameters", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-")
    plt.tight_layout()
    filename = f"{rs_key}_profiles_allsweeps.png"
    plt.savefig(outpath / filename, dpi=180)
    plt.show()

output_excel_data_for_corr = Path(outpath) / "data_for_corr.xlsx"
df_data_for_corr = pd.DataFrame(data_for_corr)
df_data_for_corr.to_excel(output_excel_data_for_corr, index=False)
print(f"'data_for_corr' saved to: {output_excel_data_for_corr}")

plt.figure(figsize=(7, 5))
im = plt.imshow(grid, origin='lower', aspect='auto')
title = "Max Thickness vs. Ri and Re"
cbar_label = "Max Thickness"
plt.xticks(range(len(Re_list)), [str(r) for r in Re_list])
plt.yticks(range(len(Ri_list)), [str(r) for r in Ri_list])
plt.xlabel("Re")
plt.ylabel("Ri")
plt.title(title)
cbar = plt.colorbar(im)
cbar.set_label(cbar_label)
plt.tight_layout()
filename = "parameter_sweep_max_thickness.png"
plt.savefig(outpath / filename, dpi=180)

#Creates velocity profiles and 1D energy spectra at various times"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# ============================================================
# USER SETTINGS
# ============================================================

INPUT_DIR = "kh_baseline_raw"
OUTPUT_DIR = "kh_output_postgraphs"

# Choose one simulation file
RUN_FILE = "baseline_results.npz"

# Times to plot from saved snapshots
SPECTRUM_TIMES = [10.0, 32.0, 60.0]
PROFILE_TIMES = [10.0, 32.0, 60.0]

# Direction for 1D spectrum:
# "x" = horizontal spectrum E(kx), usually best for KH billows
# "y" = vertical spectrum E(ky)
SPECTRUM_DIRECTION = "x"

# Optional reference slope
PLOT_MINUS_5_3 = True

# %%
def parse_run_name(filename):
    name = Path(filename).name
    # The regex pattern for sweep files includes Ri_ and Re_
    m = re.search(r"Ri_([0-9.]+)_Re_([0-9.]+)_results\.npz", name)
    if not m:
        return np.nan, np.nan
    return float(m.group(1)), float(m.group(2))


def load_run(filepath):
    data = np.load(filepath, allow_pickle=True)

    # Attempt to parse Ri, Re from filename
    Ri_from_name, Re_from_name = parse_run_name(filepath.name)

    # Use values from data if parsing from filename fails (i.e., nan)
    Ri = data.get("Ri", Ri_from_name) if np.isnan(Ri_from_name) else Ri_from_name
    Re = data.get("Re", Re_from_name) if np.isnan(Re_from_name) else Re_from_name

    run = {
        "file": Path(filepath).name,
        "Ri": Ri,
        "Re": Re,
        "X": np.asarray(data["X"], dtype=float) if "X" in data else None,
        "Y": np.asarray(data["Y"], dtype=float) if "Y" in data else None,
        "snapshot_times": np.asarray(data["snapshot_times"], dtype=float) if "snapshot_times" in data else None,
        "snapshots_u": data["snapshot_u"] if "snapshot_u" in data else None, # Corrected key
        "snapshots_w": data["snapshot_w"] if "snapshot_w" in data else None, # Corrected key
        "snapshots_omega": data["snapshot_omega"] if "snapshot_omega" in data else None, # Corrected key
    }

    # Fallback for 'snapshots_t' if 'snapshot_times' is not present
    if run["snapshot_times"] is None and "snapshots_t" in data:
        run["snapshot_times"] = np.asarray(data["snapshots_t"], dtype=float)

    # Fallback for 'omega_snapshots' if 'snapshots_omega' is not present
    if run["snapshots_omega"] is None and "omega_snapshots" in data:
        run["snapshots_omega"] = data["omega_snapshots"]

    return run


def nearest_snapshot_index(snapshot_times, target_time):
    return int(np.argmin(np.abs(snapshot_times - target_time)))
# %%
def make_wavenumbers(nx, ny, lx, ly):
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    K2 = KX**2 + KY**2
    return KX, KY, K2


def reconstruct_velocity_from_omega(omega, X, Y):
    ny, nx = omega.shape

    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    lx = dx * nx
    ly = dy * ny

    KX, KY, K2 = make_wavenumbers(nx, ny, lx, ly)

    omega_hat = np.fft.fft2(omega)
    psi_hat = np.zeros_like(omega_hat, dtype=complex)

    mask = K2 != 0.0
    psi_hat[mask] = omega_hat[mask] / K2[mask]

    u = np.fft.ifft2(1j * KY * psi_hat).real
    w = -np.fft.ifft2(1j * KX * psi_hat).real

    return u, w


def get_velocity_snapshot(run, idx):
    """
    Returns u, w for snapshot index idx.
    Uses saved u/w if available, otherwise reconstructs from omega.
    """
    if run["snapshots_u"] is not None and run["snapshots_w"] is not None:
        u = np.asarray(run["snapshots_u"][idx], dtype=float)
        w = np.asarray(run["snapshots_w"][idx], dtype=float)
        return u, w

    if run["snapshots_omega"] is not None and run["X"] is not None and run["Y"] is not None:
        omega = np.asarray(run["snapshots_omega"][idx], dtype=float)
        return reconstruct_velocity_from_omega(omega, run["X"], run["Y"])

    raise ValueError("No velocity snapshots or omega snapshots found. Save snapshots_u/snapshots_w or snapshots_omega.")

# %%
def compute_1d_energy_spectrum(u, w, X, Y, direction="x"):
    """
    Computes 1D kinetic energy spectrum.

    direction="x":
        Fourier transform along x and average over y.

    direction="y":
        Fourier transform along y and average over x.
    """

    ny, nx = u.shape

    if direction == "x":
        dx = X[0, 1] - X[0, 0]
        k = 2.0 * np.pi * np.fft.rfftfreq(nx, d=dx)

        # remove x-mean at each y to focus on fluctuations
        up = u - np.mean(u, axis=1, keepdims=True)
        wp = w - np.mean(w, axis=1, keepdims=True)

        uhat = np.fft.rfft(up, axis=1)
        what = np.fft.rfft(wp, axis=1)

        E = 0.5 * np.mean(np.abs(uhat)**2 + np.abs(what)**2, axis=0) / nx**2

    elif direction == "y":
        dy = Y[1, 0] - Y[0, 0]
        k = 2.0 * np.pi * np.fft.rfftfreq(ny, d=dy)

        # remove y-mean at each x
        up = u - np.mean(u, axis=0, keepdims=True)
        wp = w - np.mean(w, axis=0, keepdims=True)

        uhat = np.fft.rfft(up, axis=0)
        what = np.fft.rfft(wp, axis=0)

        E = 0.5 * np.mean(np.abs(uhat)**2 + np.abs(what)**2, axis=1) / ny**2

    else:
        raise ValueError("direction must be 'x' or 'y'")

    valid = (k > 0) & (E > 0)
    return k[valid], E[valid]


def plot_1d_energy_spectra(run, output_dir, times_to_plot, direction="x", plot_minus_5_3=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if run["snapshot_times"] is None:
        raise ValueError("No snapshot_times found in file.")

    X = run["X"]
    Y = run["Y"]

    plt.figure(figsize=(8, 5.5))

    reference_drawn = False

    for target_time in times_to_plot:
        idx = nearest_snapshot_index(run["snapshot_times"], target_time)
        actual_time = run["snapshot_times"][idx]

        u, w = get_velocity_snapshot(run, idx)
        k, E = compute_1d_energy_spectrum(u, w, X, Y, direction=direction)

        plt.loglog(k, E, linewidth=2, label=f"t={actual_time:.1f}")

        if plot_minus_5_3 and not reference_drawn and len(k) > 8:
            # choose middle range for reference line
            i0 = len(k) // 3
            i1 = min(len(k) - 1, i0 + len(k) // 4)

            k_ref = k[i0:i1]
            E0 = E[i0]
            C = E0 * k_ref[0]**(5.0 / 3.0)
            ref = C * k_ref**(-5.0 / 3.0)

            plt.loglog(k_ref, ref, "k--", linewidth=2, label=r"$k^{-5/3}$ reference")
            reference_drawn = True

    plt.xlabel(r"Wavenumber $k$")
    plt.ylabel(r"1D kinetic energy spectrum $E(k)$")
    plt.title(f"1D energy spectra ({direction}-direction), {run['file'].replace('_results.npz','')}")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    fname = run["file"].replace("_results.npz", f"_1D_energy_spectra_{direction}.png")
    plt.savefig(output_dir / fname, dpi=250)
    plt.close()

    print(f"Saved: {output_dir / fname}")

# %%
def plot_velocity_profiles(run, output_dir, times_to_plot):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if run["snapshot_times"] is None:
        raise ValueError("No snapshot_times found in file.")

    Y = run["Y"]
    y = Y[:, 0]

    plt.figure(figsize=(6, 6))

    for target_time in times_to_plot:
        idx = nearest_snapshot_index(run["snapshot_times"], target_time)
        actual_time = run["snapshot_times"][idx]

        u, w = get_velocity_snapshot(run, idx)

        Ubar = np.mean(u, axis=1)
        Wrms = np.sqrt(np.mean((w - np.mean(w, axis=1, keepdims=True))**2, axis=1))

        plt.plot(Ubar, y, linewidth=2, label=fr"$\bar{{u}}$, t={actual_time:.1f}")

    plt.xlabel(r"Mean horizontal velocity $\bar{u}(y)$")
    plt.ylabel("y")
    plt.title(f"Mean velocity profiles, {run['file'].replace('_results.npz','')}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = run["file"].replace("_results.npz", "_mean_velocity_profiles.png")
    plt.savefig(output_dir / fname, dpi=250)
    plt.close()

    print(f"Saved: {output_dir / fname}")

# %%
def plot_velocity_profile_with_fluctuations(run, output_dir, times_to_plot):
    """
    Creates a 1x2 figure:
      left: mean U(y)
      right: vertical RMS velocity w_rms(y)
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Y = run["Y"]
    y = Y[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)

    for target_time in times_to_plot:
        idx = nearest_snapshot_index(run["snapshot_times"], target_time)
        actual_time = run["snapshot_times"][idx]

        u, w = get_velocity_snapshot(run, idx)

        Ubar = np.mean(u, axis=1)
        Wbar = np.mean(w, axis=1, keepdims=True)
        wrms_y = np.sqrt(np.mean((w - Wbar)**2, axis=1))

        axes[0].plot(Ubar, y, linewidth=2, label=f"t={actual_time:.1f}")
        axes[1].plot(wrms_y, y, linewidth=2, label=f"t={actual_time:.1f}")

    axes[0].set_xlabel(r"$\bar{u}(y)$")
    axes[0].set_ylabel("y")
    axes[0].set_title("Mean horizontal velocity")

    axes[1].set_xlabel(r"$w_{rms}(y)$")
    axes[1].set_ylabel("y")
    axes[1].set_title("Vertical velocity fluctuations")

    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Velocity profile diagnostics, {run['file'].replace('_results.npz','')}")
    fname = run["file"].replace("_results.npz", "_velocity_profile_diagnostics.png")
    fig.savefig(output_dir / fname, dpi=250)
    plt.close(fig)

    print(f"Saved: {output_dir / fname}")

# %%
def main():
    input_path = Path(INPUT_DIR) / RUN_FILE
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Could not find: {input_path}")

    run = load_run(input_path)

    if run["X"] is None or run["Y"] is None:
        raise ValueError("X and Y arrays are required in the .npz file.")

    plot_1d_energy_spectra(
        run,
        output_dir,
        times_to_plot=SPECTRUM_TIMES,
        direction=SPECTRUM_DIRECTION,
        plot_minus_5_3=PLOT_MINUS_5_3
    )

    plot_velocity_profiles(
        run,
        output_dir,
        times_to_plot=PROFILE_TIMES
    )

    plot_velocity_profile_with_fluctuations(
        run,
        output_dir,
        times_to_plot=PROFILE_TIMES
    )


if __name__ == "__main__":
    main()

import numpy as np
from pathlib import Path
import csv
import re

# %%
INPUT_DIR = "kh_butterfly_raw"
OUTPUT_CSV = "kh_output_v4_post/flux_richardson_single_values.csv"

# Use the vertical coordinate y as height
# PE proxy = < b * y >
# This is a Boussinesq available-potential-energy proxy.
USE_ABSOLUTE_DELTA_KE = True

# %%
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def parse_run_name(filename):
    name = Path(filename).name
    m = re.search(r"Ri_([0-9.]+)_Re_([0-9.]+)_results\.npz", name)
    if not m:
        return np.nan, np.nan
    return safe_float(m.group(1)), safe_float(m.group(2))


def load_npz(path):
    return np.load(path, allow_pickle=True)


def get_b_snapshots(data):
    if "snapshots_b" in data:
        return data["snapshots_b"]
    if "b_snapshots" in data:
        return data["b_snapshots"]
    if "snapshot_b" in data:
        return data["snapshot_b"]
    return None


def compute_potential_energy_proxy(b, Y):
    """
    Boussinesq PE proxy:
        PE = < b y >

    This is not absolute gravitational potential energy,
    but it is a useful relative potential-energy diagnostic
    for comparing initial and final stratification states.
    """
    return np.mean(b * Y)


def compute_flux_richardson_for_file(path):
    data = load_npz(path)
    Ri, Re = parse_run_name(path.name)

    if "tke" not in data:
        raise ValueError("Missing tke array.")

    if "Y" not in data:
        raise ValueError("Missing Y array. Needed for PE proxy.")

    bsnaps = get_b_snapshots(data)
    if bsnaps is None:
        raise ValueError("Missing buoyancy snapshots. Need snapshots_b or b_snapshots.")

    tke = np.asarray(data["tke"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)

    b0 = np.asarray(bsnaps[0], dtype=float)
    bf = np.asarray(bsnaps[-1], dtype=float)

    KE0 = float(tke[0])
    KEf = float(tke[-1])

    PE0 = compute_potential_energy_proxy(b0, Y)
    PEf = compute_potential_energy_proxy(bf, Y)

    delta_PE = PEf - PE0

    if USE_ABSOLUTE_DELTA_KE:
        delta_KE = abs(KEf - KE0)
    else:
        delta_KE = KE0 - KEf

    if abs(delta_KE) < 1e-14:
        Rf = np.nan
    else:
        Rf = delta_PE / delta_KE

    return {
        "file": path.name,
        "Ri": Ri,
        "Re": Re,
        "KE_initial": KE0,
        "KE_final": KEf,
        "PE_initial_proxy": PE0,
        "PE_final_proxy": PEf,
        "delta_PE": delta_PE,
        "delta_KE": delta_KE,
        "Rf": Rf,
    }

# %%
def main():
    input_dir = Path(INPUT_DIR)
    output_csv = Path(OUTPUT_CSV)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for path in sorted(input_dir.glob("*_results.npz")):
        try:
            row = compute_flux_richardson_for_file(path)
            rows.append(row)
            print(f"{path.name}: Rf = {row['Rf']:.6f}")
        except Exception as e:
            print(f"Skipping {path.name}: {e}")

    if not rows:
        print("No valid files processed.")
        return

    keys = list(rows[0].keys())
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    main()

OUTPUT_EXCEL = "kh_output_v4_post/transition_times_sweeps.xlsx"

transition_times_data = []

for (ri, re), results_dict in all_sweep_results.items():
    transition_time = results_dict.get('transition_time', np.nan)
    transition_times_data.append({"Ri": ri, "Re": re, "Transition_Time": transition_time})

df_transition_times = pd.DataFrame(transition_times_data)

df_transition_times.to_excel(OUTPUT_EXCEL, index=False)

print(f"Transition times extracted and saved to: {OUTPUT_EXCEL}")
display(df_transition_times)

print(f"Transition time for the baseline run: {results['summary']['transition_time']}")

"""Creating an animated GIF from One Simulation Result

This will create an animated GIF of the vorticity field for the baseline simulation, showing its evolution over time.
"""
# %%
butterfly_output_gif_name = Path(cfg.outdir) / f"vorticity_animation_{cfg.label}.gif"
field_to_animate_bufferfly = 'b' # Can be 'omega', 'b', 'u', or 'w'

# Call the function to create the GIF for the baseline run
create_snapshot_gif(
    results,
    field_to_animate_bufferfly,
    str(butterfly_output_gif_name), # Convert Path object to string for os.path.join
    fps=10, # Adjust frames per second as desired
    cmap='viridis' if field_to_animate_bufferfly == 'b' else 'RdBu_r',
    title=f"Evolution of {field_to_animate_bufferfly.capitalize()} (Butterfly)"
)