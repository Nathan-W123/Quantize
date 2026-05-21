import numpy as np
from scipy.stats import skew, kurtosis, gaussian_kde
from scipy.signal import find_peaks
from backend.spectral.spectral import SpectralEngine
from backend.internal.internal_fit import InternalCoordinateSet

def detect_modality(data):
    """
    Heuristic to detect multi-modality by counting peaks in a Gaussian KDE.
    Returns (num_peaks, is_multimodal).
    """
    if len(data) < 20:
        return 1, False

    # Skip near-constant parameters to avoid KDE singular matrix errors
    if np.ptp(data) < 1e-10:
        return 1, False

    try:
        kde = gaussian_kde(data)
        x = np.linspace(np.min(data), np.max(data), 200)
        y = kde(x)
        # Peaks must be > 10% of max density with moderate prominence to count
        peaks, _ = find_peaks(y, height=np.max(y) * 0.1, prominence=np.max(y) * 0.05)
        return len(peaks), len(peaks) > 1
    except Exception:
        return 1, False

def analyze_coordinate_distribution(cloud_xyz, elems, use_dihedrals=False):
    """
    Converts a cloud of Cartesian geometries into statistics for 
    internal coordinates (bonds, angles, dihedrals).
    """
    # Initialize the topology using the first sample
    ic_set = InternalCoordinateSet(cloud_xyz[0], elems, use_dihedrals=use_dihedrals)
    names = ic_set.active_names()
    
    # Project all Cartesians in the cloud to Internals
    # Shape: (n_samples, n_coordinates)
    q_cloud = np.array([ic_set.active_values(xyz) for xyz in cloud_xyz])
    
    results = []
    for i, name in enumerate(names):
        data = q_cloud[:, i]
        
        num_peaks, is_multimodal = detect_modality(data)
        
        # Handle units (Internals are in Angstrom/Radians)
        is_angle = "angle" in name or "dihedral" in name
        if is_angle:
            data = np.degrees(data)
            unit = "deg"
        else:
            unit = "Ã…"
            
        results.append({
            "name": name,
            "unit": unit,
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data)),
            "ci_95": [float(np.percentile(data, 2.5)), float(np.percentile(data, 97.5))],
            "skew": float(skew(data)),
            "kurtosis": float(kurtosis(data)),
            "num_peaks": num_peaks,
            "is_multimodal": is_multimodal
        })
        
    # Parameter correlation matrix
    corr = np.corrcoef(q_cloud.T)
    
    return results, corr, names

def print_distribution_summary(stats):
    """Prints the posterior summary table with mean, median, and 95% CI."""
    print("\n" + "="*100)
    print(f"{'Coordinate':<30} {'Mean':>10} {'Median':>10} {'StdDev':>10} {'95% Credible Interval':>25} {'Skew'} {'Peaks'}")
    print("-" * 100)
    for s in stats:
        ci_str = f"[{s['ci_95'][0]:.4f}, {s['ci_95'][1]:.4f}]"
        mod_mark = "!" if s['is_multimodal'] else " "
        print(f"{s['name']:<30} {s['mean']:10.4f} {s['median']:10.4f} {s['std']:10.4f} {ci_str:>25} {s['skew']:+6.2f} {mod_mark}{s['num_peaks']}")
    print("="*100)
    if any(s['is_multimodal'] for s in stats):
        print("  (!) WARNING: Multi-modality detected in one or more coordinates.")

def posterior_predictive_check(cloud_xyz, spectral_engine: SpectralEngine):
    """
    Posterior Predictive Check (PPC): Verify that the sampled geometries 
    actually replicate the observed experimental data.
    """
    all_rmses = []
    for xyz in cloud_xyz:
        _, residual_mhz = spectral_engine.stacked_unweighted(xyz)
        rmse = np.sqrt(np.mean(residual_mhz**2))
        all_rmses.append(rmse)
    
    all_rmses = np.array(all_rmses)
    return {
        "mean_rmse_mhz": np.mean(all_rmses),
        "median_rmse_mhz": np.median(all_rmses),
        "std_rmse_mhz": np.std(all_rmses),
        "ci_95_rmse_mhz": [np.percentile(all_rmses, 2.5), np.percentile(all_rmses, 97.5)]
    }

def print_ppc_summary(ppc_stats):
    """Prints the results of the Posterior Predictive Check."""
    print("\n" + "="*60)
    print("  POSTERIOR PREDICTIVE CHECK (Spectral Residuals)")
    print("="*60)
    print(f"  Mean RMS Residual   : {ppc_stats['mean_rmse_mhz']:.4f} MHz")
    print(f"  Median RMS Residual : {ppc_stats['median_rmse_mhz']:.4f} MHz")
    print(f"  Std Dev             : {ppc_stats['std_rmse_mhz']:.4f} MHz")
    ci = ppc_stats['ci_95_rmse_mhz']
    print(f"  95% Predictive CI   : [{ci[0]:.4f}, {ci[1]:.4f}] MHz")
    print("="*60)