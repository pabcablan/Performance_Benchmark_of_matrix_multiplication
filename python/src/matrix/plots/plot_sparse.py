import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np
from matplotlib.ticker import FuncFormatter

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df.groupby(['Algorithm', 'Size', 'Sparsity']).agg({'TimeSeconds': 'mean', 'MemoryMB': 'mean'}).reset_index()

def plot_pure(df, output_dir):
    sizes = [256, 512, 1024, 2048]
    sparsities = [0.5, 0.7, 0.9, 0.99]
    data = df[df['Algorithm'] == 'CSR-Pure']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for size in sizes:
        subset = data[data['Size'] == size]
        sparsity_pct = subset['Sparsity'] * 100
        ax1.plot(sparsity_pct, subset['TimeSeconds'], marker='o', label=f'{size}×{size}', linewidth=2)
    
    ax1.set_xlabel('Sparsity (%)')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Time vs Sparsity (Multiple Sizes)')
    ax1.set_yscale('log')
    ax1.set_xticks([50, 70, 90, 95, 99])
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3g}'))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for sparsity in sparsities:
        subset = data[data['Sparsity'] == sparsity]
        ax2.plot(subset['Size'], subset['TimeSeconds'], marker='s', label=f'{sparsity*100:.0f}% sparse', linewidth=2)
    
    ax2.set_xlabel('Matrix Size (n×n)')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Time vs Size (Multiple Sparsities)')
    ax2.set_yscale('log')
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(data['Size'].unique())
    ax2.set_xticklabels(data['Size'].unique())
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3g}'))
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.suptitle('CSR-Pure Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sparse_pure_analysis.png', dpi=300)
    plt.close()

def plot_scipy(df, output_dir):
    sizes = [256, 512, 1024, 2048]
    sparsities = [0.5, 0.7, 0.9, 0.99]
    data = df[df['Algorithm'] == 'CSR-SciPy']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for size in sizes:
        subset = data[data['Size'] == size]
        sparsity_pct = subset['Sparsity'] * 100
        ax1.plot(sparsity_pct, subset['TimeSeconds'], marker='o', label=f'{size}×{size}', linewidth=2)
    
    ax1.set_xlabel('Sparsity (%)')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Time vs Sparsity (Multiple Sizes)')
    ax1.set_yscale('log')
    ax1.set_xticks([50, 70, 90, 95, 99])
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3g}'))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for sparsity in sparsities:
        subset = data[data['Sparsity'] == sparsity]
        ax2.plot(subset['Size'], subset['TimeSeconds'], marker='s', label=f'{sparsity*100:.0f}% sparse', linewidth=2)
    
    ax2.set_xlabel('Matrix Size (n×n)')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Time vs Size (Multiple Sparsities)')
    ax2.set_yscale('log')
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(data['Size'].unique())
    ax2.set_xticklabels(data['Size'].unique())
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3g}'))
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.suptitle('CSR-SciPy Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sparse_scipy_analysis.png', dpi=300)
    plt.close()

def plot_comparison(df, output_dir):
    sparsity = 0.9
    sizes = [256, 512, 1024, 2048]
    data = df[(df['Sparsity'] == sparsity) & (df['Size'].isin(sizes))]
    
    pure = data[data['Algorithm'] == 'CSR-Pure'].set_index('Size')['TimeSeconds']
    scipy = data[data['Algorithm'] == 'CSR-SciPy'].set_index('Size')['TimeSeconds']
    
    x = np.arange(len(sizes))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, pure.values, width, label='CSR-Pure', color='#e74c3c')
    plt.bar(x + width/2, scipy.values, width, label='CSR-SciPy', color='#3498db')
    
    plt.xlabel('Matrix Size (n×n)')
    plt.ylabel('Time (seconds)')
    plt.title(f'CSR-Pure vs CSR-SciPy (Sparsity {sparsity*100:.0f}%)')
    plt.xticks(x, sizes)
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3g}'))
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sparse_comparison.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_sparse.py <csv_file> <output_directory>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating plots from: {csv_file}")
    
    df = load_data(csv_file)
    
    plot_pure(df, output_dir)
    plot_scipy(df, output_dir)
    plot_comparison(df, output_dir)
    
    print(f"\nAll 3 plots saved to {output_dir}")