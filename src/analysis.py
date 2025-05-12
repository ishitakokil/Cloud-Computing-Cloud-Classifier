from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Logger configuration
logger = logging.getLogger("eda_plotter")

def save_figures(df: pd.DataFrame, output_dir: Path, target_col: str = "cloud_type") -> None:
    """
    Generate and save EDA plots: correlation heatmap, individual histograms, and class-wise histograms.

    Args:
        df: Input DataFrame.
        output_dir: Directory to save figures.
        target_col: Column name for class labels (default: 'cloud_type').
    """
    try:
        logger.info("Generating EDA figures.")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap.png")
        plt.close()

        # 2. Individual histograms
        for col in df.select_dtypes(include="number").columns:
            if col == target_col:
                continue
            plt.figure()
            sns.histplot(df[col], kde=True)
            plt.title(f"Histogram: {col}")
            plt.tight_layout()
            plt.savefig(output_dir / f"{col}_hist.png")
            plt.close()

        # 3. Class-wise histograms
        numeric_cols = [col for col in df.select_dtypes(include="number").columns if col != target_col]
        n_cols = 2
        n_rows = (len(numeric_cols) + 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
        axes = axes.flatten()
        target = df[target_col]

        for i, col in enumerate(numeric_cols):
            ax = axes[i]
            ax.hist(
                [df[target == 0][col].dropna(), df[target == 1][col].dropna()],
                label=["Class 0", "Class 1"],
                bins=30,
                alpha=0.7,
            )
            ax.set_title(f"{col} by Class")
            ax.set_xlabel(" ".join(col.split("_")).capitalize())
            ax.set_ylabel("Count")
            ax.legend()

        # Remove unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        fig.savefig(output_dir / "all_features_by_class.png")
        plt.close()

        logger.info("EDA figures saved to %s", output_dir)

    except Exception as e:
        logger.exception("Failed to generate and save EDA figures.")
        raise RuntimeError(f"EDA plot generation failed: {e}")
