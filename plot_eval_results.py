import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load evaluation data
df = pd.read_csv("results/evaluation_summary.csv")

# Ensure output directory exists
os.makedirs("results/plots", exist_ok=True)

# Standardized color palette
palette = sns.color_palette("Set2")

# Metrics to plot
metrics = ["Accuracy", "F1 Score", "Real Acc", "Fake Recall"]
metric_titles = {
    "Accuracy": "Model Accuracy Comparison",
    "F1 Score": "F1 Score Comparison",
    "Real Acc": "Real Image Accuracy Comparison",
    "Fake Recall": "Fake Image Recall Comparison"
}

# Generate plots
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Model", y=metric, hue="Test Set", palette=palette)
    plt.title(metric_titles[metric])
    plt.ylabel(metric)
    plt.xticks(rotation=15)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    filename = f"results/plots/{metric.lower().replace(' ', '_')}_comparison.png"
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()

print("✅ All plots generated successfully.")
