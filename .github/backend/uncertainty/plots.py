import os
import numpy as np
import matplotlib.pyplot as plt
from backend.internal.internal_fit import InternalCoordinateSet

def plot_coordinate_distributions(cloud_xyz, elems, output_dir, use_dihedrals=False):
    """Generates histogram plots for every active internal coordinate."""
    os.makedirs(output_dir, exist_ok=True)
    ic_set = InternalCoordinateSet(cloud_xyz[0], elems, use_dihedrals=use_dihedrals)
    names = ic_set.active_names()
    q_cloud = np.array([ic_set.active_values(xyz) for xyz in cloud_xyz])
    
    for i, name in enumerate(names):
        data = q_cloud[:, i]
        if "angle" in name or "dihedral" in name:
            data = np.degrees(data)
            unit = "deg"
        else:
            unit = "Ã…"
            
        plt.figure(figsize=(6, 4))
        plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        mu, sigma = np.mean(data), np.std(data)
        plt.axvline(mu, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mu:.4f}')
        plt.title(f"Distribution of {name}")
        plt.xlabel(f"Value ({unit})")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        
        safe_name = name.replace(" ", "_").replace("-", "_").replace("/", "_")
        plt.savefig(os.path.join(output_dir, f"dist_{safe_name}.png"), dpi=150)
        plt.close()

def plot_mcmc_traces(samples_xyz, elems, output_dir, use_dihedrals=False):
    """Generates trace plots to diagnose MCMC convergence."""
    os.makedirs(output_dir, exist_ok=True)
    ic_set = InternalCoordinateSet(samples_xyz[0], elems, use_dihedrals=use_dihedrals)
    names = ic_set.active_names()
    q_samples = np.array([ic_set.active_values(xyz) for xyz in samples_xyz])
    
    n = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2 * n), sharex=True)
    if n == 1: axes = [axes]
    
    for i, name in enumerate(names):
        data = q_samples[:, i]
        if "angle" in name or "dihedral" in name:
            data = np.degrees(data)
        axes[i].plot(data, lw=0.5, color='teal')
        axes[i].set_ylabel(name, fontsize=9)
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("MCMC Step (after burn-in)")
    plt.suptitle("MCMC Parameter Traces", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "mcmc_traces.png"), dpi=200)
    plt.close()

def plot_parameter_corner(cloud_xyz, elems, output_dir, use_dihedrals=False):
    """Generates a correlation matrix (corner plot) for all parameters."""
    ic_set = InternalCoordinateSet(cloud_xyz[0], elems, use_dihedrals=use_dihedrals)
    names = ic_set.active_names()
    q_cloud = np.array([ic_set.active_values(xyz) for xyz in cloud_xyz])
    n = len(names)
    
    fig, axes = plt.subplots(n, n, figsize=(min(20, 2.5 * n), min(20, 2.5 * n)))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # Diagonal: Histogram
                data = q_cloud[:, i]
                if "angle" in names[i] or "dihedral" in names[i]: data = np.degrees(data)
                ax.hist(data, bins=20, color='gray', alpha=0.6)
            elif i > j:
                # Off-diagonal: Scatter
                x = q_cloud[:, j]
                y = q_cloud[:, i]
                if "angle" in names[j] or "dihedral" in names[j]: x = np.degrees(x)
                if "angle" in names[i] or "dihedral" in names[i]: y = np.degrees(y)
                ax.scatter(x, y, s=2, alpha=0.2, color='navy')
            else:
                ax.axis('off')
            
            if j == 0: ax.set_ylabel(names[i], fontsize=8)
            if i == n - 1: ax.set_xlabel(names[j], fontsize=8)
            ax.tick_params(labelsize=7)
            
    plt.suptitle("Parameter Correlation Matrix (Corner Plot)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "corner_plot.png"), dpi=200)
    plt.close()

def plot_autocorrelations(samples_xyz, elems, output_dir, max_lag=100):
    """Plots autocorrelation functions (ACF) to diagnose MCMC mixing."""
    os.makedirs(output_dir, exist_ok=True)
    ic_set = InternalCoordinateSet(samples_xyz[0], elems)
    names = ic_set.active_names()
    q_samples = np.array([ic_set.active_values(xyz) for xyz in samples_xyz])
    
    n = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2 * n), sharex=True)
    if n == 1: axes = [axes]
    
    for i, name in enumerate(names):
        data = q_samples[:, i]
        data = data - np.mean(data)
        
        acf = [1.0]
        for lag in range(1, min(max_lag, len(data))):
            if np.var(data) > 1e-12:
                acf.append(np.corrcoef(data[:-lag], data[lag:])[0, 1])
            else:
                acf.append(0.0)
            
        axes[i].bar(range(len(acf)), acf, color='gray', alpha=0.7)
        axes[i].set_ylabel(name, fontsize=8)
        axes[i].set_ylim(-1, 1)
        
    axes[-1].set_xlabel("Lag")
    plt.suptitle("MCMC Autocorrelation Functions (ACF)", fontsize=14)
    plt.savefig(os.path.join(output_dir, "autocorrelations.png"), dpi=150)
    plt.close()