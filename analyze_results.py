import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Target directory containing output CSV files
DATA_DIR = "D:/results"  # Change to "./results" if your files reside in the local project directory


def load_and_label_data():
    all_data = []

    # Experimental condition mapping: Pattern to (Browser, Framework) tuple
    conditions = {
        "c1_run*.csv": ("Chrome", "TensorFlow.js (WebGL)"),
        "c3_run*.csv": ("Edge", "TensorFlow.js (WebGL)"),
        "c5_run*.csv": ("Chrome", "ONNX Runtime (WASM)"),
        "c7_run*.csv": ("Edge", "ONNX Runtime (WASM)")
    }

    for pattern, (browser, framework) in conditions.items():
        files = glob.glob(os.path.join(DATA_DIR, pattern))
        for f in files:
            df = pd.read_csv(f)
            df['Browser'] = browser
            df['Framework'] = framework
            df['Run_File'] = os.path.basename(f)
            all_data.append(df)

    if not all_data:
        print(f"Error: No CSV benchmark files found at location '{DATA_DIR}'. Please check the file directory path.")
        return None

    return pd.concat(all_data, ignore_index=True)


def run_statistical_analysis():
    df = load_and_label_data()
    if df is None:
        return

    print("\n================ DATASET OVERVIEW ================")
    print(f"Total Data Points Processed: {len(df)}")
    print(df.groupby(['Browser', 'Framework']).size().reset_index(name='Sample_Count'))

    # 1. Compute Factorial Summary Statistics
    summary = df.groupby(['Browser', 'Framework']).agg(
        Mean_Latency=('Latency_ms', 'mean'),
        Std_Latency=('Latency_ms', 'std'),
        Mean_Heap_MB=('Used_Heap_MB', 'mean'),
        Max_Heap_MB=('Used_Heap_MB', 'max'),
        Min_Heap_MB=('Used_Heap_MB', 'min')
    ).reset_index()

    print("\n================ SUMMARY STATISTICS ================")
    print(summary.to_string(index=False))
    summary.to_csv("factorial_summary_stats.csv", index=False)

    # 2. Execute Two-Way ANOVA Tests
    print("\n================ TWO-WAY ANOVA: INFERENCE LATENCY ================")
    model_lat = ols('Latency_ms ~ C(Browser) + C(Framework) + C(Browser):C(Framework)', data=df).fit()
    anova_lat = sm.stats.anova_lm(model_lat, typ=2)
    print(anova_lat)

    print("\n================ TWO-WAY ANOVA: HEAP MEMORY FOOTPRINT ================")
    model_heap = ols('Used_Heap_MB ~ C(Browser) + C(Framework) + C(Browser):C(Framework)', data=df).fit()
    anova_heap = sm.stats.anova_lm(model_heap, typ=2)
    print(anova_heap)

    # 3. Generate Visualizations for Research Publication
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Inference Latency by Browser and Framework
    sns.boxplot(
        ax=axes[0], 
        data=df, 
        x='Browser', 
        y='Latency_ms', 
        hue='Framework', 
        palette="Set2"
    )
    axes[0].set_title('Inference Latency by Browser & Runtime (ms)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Latency (ms)')

    # Plot 2: Memory Footprint Stability Across Cycles
    sns.lineplot(
        ax=axes[1], 
        data=df, 
        x='Cycle', 
        y='Used_Heap_MB', 
        hue='Framework', 
        style='Browser', 
        errorbar='sd'
    )
    axes[1].set_title('Heap Memory Footprint Across Cycles', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Used JS Heap (MB)')

    plt.tight_layout()
    plt.savefig("factorial_analysis_charts.png", dpi=300)
    print("\n[SUCCESS] Statistical summary table and analytical graphs successfully exported.")


if __name__ == "__main__":
    run_statistical_analysis()