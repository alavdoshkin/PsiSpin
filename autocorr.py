# Copyright (c) 2025 Alexander Avdoshkin, Max Geier, Massachusetts Institute of Technology, MA, USA
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt

def load_energy_real(csv_path):
    n = 10000
    energies = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "energy" not in reader.fieldnames:
            raise ValueError(f"'energy' column not found. Columns: {reader.fieldnames}")
        for row in itertools.islice(reader, n):
            z = complex(row["energy"].strip())
            energies.append(z.real)
    return np.asarray(energies, dtype=float)

def autocorr_fft(x):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    f = np.fft.rfft(x, n=2*n)
    acf = np.fft.irfft(f * np.conjugate(f), n=2*n)[:n]
    acf /= acf[0]
    return acf

def integrated_autocorr_time(x, c=5.0):
    rho = autocorr_fft(x)
    tau = 0.5
    W = 1
    while W < len(rho) - 1:
        tau = 0.5 + float(rho[1:W+1].sum())
        new_W = int(np.ceil(c * tau))
        if new_W <= W:
            break
        W = new_W
    W = min(W, len(rho) - 1)

    # keeping your hard override
    W = 50
    tau = 0.5 + float(rho[1:W+1].sum())
    return float(tau), int(W)

def z_critical(confidence=0.95):
    if abs(confidence - 0.90) < 1e-12:
        return 1.6448536269514722
    if abs(confidence - 0.95) < 1e-12:
        return 1.959963984540054
    if abs(confidence - 0.99) < 1e-12:
        return 2.5758293035489004
    raise ValueError("Only 0.90/0.95/0.99 supported without SciPy.")

def mean_se_ci_autocorr(x, confidence=0.95, c=5.0):
    x = np.asarray(x, dtype=float)
    N = x.size
    if N < 2:
        raise ValueError("Need at least 2 samples.")
    mean = float(x.mean())
    s2 = float(x.var(ddof=1))
    tau_int, W = integrated_autocorr_time(x, c=c)
    neff = N / (2.0 * tau_int) if tau_int > 0 else float("inf")
    se_mean = math.sqrt((2.0 * tau_int * s2) / N) if tau_int > 0 else 0.0
    crit = z_critical(confidence)
    half = crit * se_mean
    return {
        "N": N,
        "mean": mean,
        "sample_std": math.sqrt(s2),
        "tau_int": tau_int,
        "window_W": W,
        "neff": neff,
        "se_mean": se_mean,
        "confidence": confidence,
        "ci_low": mean - half,
        "ci_high": mean + half,
    }

# ---------------- NEW: exponential fit for ACF ----------------

def fit_acf_exponential(rho, t_min=1, t_max=None, min_rho=1e-3, max_points=None):
    """
    Fit rho(t) ~ A * exp(-t/tau) using linear regression on log(rho).
    Uses only points where rho>0 and rho>=min_rho.
    Returns dict with A, tau_exp, slope, intercept, r2, used_lags.

    Args:
      rho: array-like ACF with rho[0]=1
      t_min: start lag (>=1)
      t_max: end lag (inclusive). If None, uses len(rho)-1
      min_rho: ignore points with rho < min_rho (noise floor)
      max_points: optionally cap number of points (take earliest ones)
    """
    rho = np.asarray(rho, dtype=float)
    n = rho.size
    if t_max is None:
        t_max = n - 1
    t_max = min(t_max, n - 1)

    lags = np.arange(t_min, t_max + 1)
    vals = rho[t_min : t_max + 1]

    mask = (vals > 0.0) & (vals >= float(min_rho)) & np.isfinite(vals)
    lags = lags[mask]
    vals = vals[mask]

    if lags.size < 3:
        return {
            "ok": False,
            "reason": f"Not enough positive ACF points above min_rho={min_rho} in [{t_min},{t_max}]",
        }

    # Optional: use earliest points (often cleaner)
    if max_points is not None and lags.size > max_points:
        lags = lags[:max_points]
        vals = vals[:max_points]

    y = np.log(vals)
    x = lags.astype(float)

    # ordinary least squares for y = b + m x
    x_mean = x.mean()
    y_mean = y.mean()
    Sxx = np.sum((x - x_mean) ** 2)
    Sxy = np.sum((x - x_mean) * (y - y_mean))
    m = Sxy / Sxx
    b = y_mean - m * x_mean

    # decay time tau = -1/m (need m<0)
    if m >= 0:
        return {"ok": False, "reason": f"Non-decaying fit slope m={m:.3g} (ACF too noisy / bad window?)"}

    tau_exp = -1.0 / m
    A = float(np.exp(b))

    # R^2 in log-space
    y_hat = b + m * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "ok": True,
        "A": A,
        "tau_exp": float(tau_exp),
        "slope": float(m),
        "intercept": float(b),
        "r2_log": float(r2),
        "used_lags": lags,
        "used_rho": vals,
        "fit_range": (int(lags[0]), int(lags[-1])),
        "min_rho": float(min_rho),
    }

def plot_acf_with_exp_fit(
    x,
    max_lag=100,
    c=5.0,
    outpath="acf_fit.png",
    fit_t_min=1,
    fit_t_max=200,
    fit_min_rho=1e-3,
    fit_max_points=None,
    title="ACF with exponential fit",
):
    rho = autocorr_fft(x)
    L = min(max_lag, len(rho) - 1)
    lags = np.arange(L + 1)
    rhoL = rho[: L + 1]

    tau_int, W = integrated_autocorr_time(x, c=c)
    fit = fit_acf_exponential(rho, t_min=fit_t_min, t_max=fit_t_max, min_rho=fit_min_rho, max_points=fit_max_points)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(lags, rhoL, lw=1.5, label="ACF ρ(t)")
    ax.axhline(0.0, lw=1.0)
    #ax.axvline(W, ls="--", lw=1.2, label=f"window W={W}")
    ax.set_xlabel("lag t")
    ax.set_ylabel("ρ(t)")
    ax.set_xlim(0, L)
    ax.grid(True, alpha=0.3)

    subtitle = f"τ_int ≈ {tau_int:.3f} (W={W})"

    if fit.get("ok", False):
        A = fit["A"]
        tau_exp = fit["tau_exp"]
        t0, t1 = fit["fit_range"]
        t_fit = np.arange(t0, min(t1, L) + 1)
        ax.plot(t_fit, A * np.exp(-t_fit / tau_exp), lw=2.0, ls="-.", label=f"fit: A exp(-t/τ), τ={tau_exp:.2f}")
        ax.axvspan(t0, t1, alpha=0.08, label=f"fit range [{t0},{t1}]")
        subtitle += f", τ_exp ≈ {tau_exp:.3f} (R²_log={fit['r2_log']:.3f})"
    else:
        subtitle += f", exp fit failed: {fit.get('reason','unknown')}"

    ax.set_title(f"{title}\n{subtitle}")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath, fit

if __name__ == "__main__":
    # Edit this to point at an inference_stats.csv produced by run_soc.py / run_zeeman.py / run_pbc.py.
    file = "results/my_run/inference_stats.csv"

    energies_real = load_energy_real(file)

    stats = mean_se_ci_autocorr(energies_real, confidence=0.95, c=5.0)
    print(f"N = {stats['N']}")
    print(f"mean(E) = {stats['mean']:.8f}")
    print(f"std(E)  = {stats['sample_std']:.8f}")
    print(f"tau_int ≈ {stats['tau_int']:.3f} steps (window W={stats['window_W']})")
    print(f"Neff    ≈ {stats['neff']:.1f}")
    print(f"SE(mean)≈ {stats['se_mean']:.8f}")
    print(f"{int(stats['confidence']*100)}% CI ≈ [{stats['ci_low']:.8f}, {stats['ci_high']:.8f}]")

    # Compute ACF and fit exponential decay time
    rho = autocorr_fft(energies_real)
    fit = fit_acf_exponential(rho, t_min=1, t_max=200, min_rho=1e-3, max_points=None)
    if fit.get("ok", False):
        print("\nExponential ACF fit: rho(t) ~ A exp(-t/tau)")
        print(f"A        ≈ {fit['A']:.6f}")
        print(f"tau_exp  ≈ {fit['tau_exp']:.3f} steps")
        print(f"fit lags  = {fit['fit_range'][0]} .. {fit['fit_range'][1]}")
        print(f"R^2 (log) ≈ {fit['r2_log']:.4f}")
    else:
        print("\nExponential ACF fit failed:", fit.get("reason", "unknown"))

    # Plot ACF + fit overlay
    outpath, fit2 = plot_acf_with_exp_fit(
        energies_real,
        max_lag=30,
        c=5.0,
        outpath="acf_fit.png",
        fit_t_min=1,
        fit_t_max=5,     # adjust if you want
        fit_min_rho=1e-3,  # adjust noise floor
        fit_max_points=None,
        title="Energy ACF with exponential fit",
    )
    print(f"\nSaved ACF+fit plot to: {outpath}")
