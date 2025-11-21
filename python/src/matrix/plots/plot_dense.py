import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np
from matplotlib.ticker import FuncFormatter

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df.groupby(['Algorithm', 'Size']).agg({'TimeSeconds': 'mean', 'MemoryMB': 'mean'}).reset_index()

def plot_python_pure(df, output_dir):
    algorithms = ['Standard', 'Row-Oriented', 'Tiled-32', 'Tiled-64', 'Strassen']
    data = df[df['Algorithm'].isin(algorithms)]
    
    plt.figure()
    for algo in algorithms:
        subset = data[data['Algorithm'] == algo]
        plt.plot(subset['Size'], subset['TimeSeconds'], marker='o', label=algo, linewidth=2)
    
    plt.xlabel('Matrix Size (n×n)')
    plt.ylabel('Time (seconds)')
    plt.title('Python Pure - Time vs Size')
    plt.yscale('log')
    plt.xscale('log', base=2)
    plt.xticks(data['Size'].unique(), data['Size'].unique())
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3g}'))
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/python_pure_time_dense.png', dpi=300)
    plt.close()

def plot_numpy(df, output_dir):
    algorithms = ['NumPy-builtin', 'NumPy-matmul', 'NumPy-Tiled-64', 'NumPy-Strassen']
    data = df[df['Algorithm'].isin(algorithms)]
    
    plt.figure()
    for algo in algorithms:
        subset = data[data['Algorithm'] == algo]
        plt.plot(subset['Size'], subset['TimeSeconds'], marker='s', label=algo, linewidth=2)
    
    plt.xlabel('Matrix Size (n×n)')
    plt.ylabel('Time (seconds)')
    plt.title('NumPy - Time vs Size')
    plt.yscale('log')
    plt.xscale('log', base=2)
    plt.xticks(data['Size'].unique(), data['Size'].unique())
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3g}'))
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/numpy_time_dense.png', dpi=300)
    plt.close()

def plot_comparison(df, output_dir):
    python_best = df[df['Algorithm'] == 'Strassen'].set_index('Size')['TimeSeconds']
    numpy_best = df[df['Algorithm'] == 'NumPy-builtin'].set_index('Size')['TimeSeconds']
    
    sizes = python_best.index.values
    x = np.arange(len(sizes))
    width = 0.35
    
    plt.figure()
    plt.bar(x - width/2, python_best.values, width, label='Python Pure (Strassen)', color='#e74c3c')
    plt.bar(x + width/2, numpy_best.values, width, label='NumPy (builtin)', color='#3498db')
    
    plt.xlabel('Matrix Size (n×n)')
    plt.ylabel('Time (seconds)')
    plt.title('Best Python Pure vs Best NumPy')
    plt.xticks(x, sizes)
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3g}'))
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/python_vs_numpy_dense.png', dpi=300)
    plt.close()

def plot_memory(df, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    python_algos = ['Standard', 'Row-Oriented', 'Tiled-64', 'Strassen']
    for algo in python_algos:
        subset = df[df['Algorithm'] == algo]
        lw = 3 if algo == 'Strassen' else 2
        ax1.plot(subset['Size'], subset['MemoryMB'], marker='o', label=algo, linewidth=lw)
    
    ax1.set_xlabel('Matrix Size (n×n)')
    ax1.set_ylabel('Memory (MB)')
    ax1.set_title('Python Pure - Memory Usage')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xticks(subset['Size'].unique())
    ax1.set_xticklabels(subset['Size'].unique())
    ax1.set_yticks([50, 100, 200, 500, 1000])
    ax1.set_yticklabels(['50', '100', '200', '500', '1000'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    numpy_algos = ['NumPy-builtin', 'NumPy-matmul', 'NumPy-Tiled-64', 'NumPy-Strassen']
    for algo in numpy_algos:
        subset = df[df['Algorithm'] == algo]
        lw = 3 if 'Strassen' in algo else 2
        ax2.plot(subset['Size'], subset['MemoryMB'], marker='s', label=algo, linewidth=lw)
    
    ax2.set_xlabel('Matrix Size (n×n)')
    ax2.set_ylabel('Memory (MB)')
    ax2.set_title('NumPy - Memory Usage')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.set_xticks(subset['Size'].unique())
    ax2.set_xticklabels(subset['Size'].unique())
    ax2.set_yticks([60, 100, 150, 200])
    ax2.set_yticklabels(['60', '100', '150', '200'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/memory_comparison_dense.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_dense.py <csv_file> <output_directory>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating plots from: {csv_file}")
    
    df = load_data(csv_file)
    
    plot_python_pure(df, output_dir)
    plot_numpy(df, output_dir)
    plot_comparison(df, output_dir)
    plot_memory(df, output_dir)
    
    print(f"\nAll 4 plots saved to {output_dir}")