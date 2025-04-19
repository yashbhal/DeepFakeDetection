import pandas as pd
from pathlib import Path

data = [
    {
        "Model": "Original (Pretrained)",
        "Training": "GANs only",
        "Test Set": "Original",
        "Accuracy": 0.509,
        "F1 Score": 0.071,
        "Real Acc": 0.987,
        "Fake Recall": 0.038
    },
    {
        "Model": "Original (Pretrained)",
        "Training": "GANs only",
        "Test Set": "Color-Jittered",
        "Accuracy": 0.509,
        "F1 Score": 0.071,
        "Real Acc": 0.987,
        "Fake Recall": 0.038
    },
    {
        "Model": "Fine-Tuned (Original Data)",
        "Training": "Deepfake-vs-Real",
        "Test Set": "Original",
        "Accuracy": 0.711,
        "F1 Score": 0.775,
        "Real Acc": 0.426,
        "Fake Recall": 0.991
    },
    {
        "Model": "Fine-Tuned (Original Data)",
        "Training": "Deepfake-vs-Real",
        "Test Set": "Color-Jittered",
        "Accuracy": 0.711,
        "F1 Score": 0.775,
        "Real Acc": 0.426,
        "Fake Recall": 0.991
    },
    {
        "Model": "Fine-Tuned (Color-Jittered)",
        "Training": "Jittered Deepfake-vs-Real",
        "Test Set": "Color-Jittered",
        "Accuracy": 0.737,
        "F1 Score": 0.790,
        "Real Acc": 0.489,
        "Fake Recall": 0.981
    },
    {
        "Model": "Fine-Tuned (Color-Jittered)",
        "Training": "Jittered Deepfake-vs-Real",
        "Test Set": "Original",
        "Accuracy": 0.737,
        "F1 Score": 0.790,
        "Real Acc": 0.489,
        "Fake Recall": 0.981
    },
]

df = pd.DataFrame(data)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/evaluation_summary.csv", index=False)
print("✅ CSV regenerated at results/evaluation_summary.csv")
