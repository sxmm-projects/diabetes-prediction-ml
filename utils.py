import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    return plt


def plot_feature_importance(importance):
    plt.figure(figsize=(6, 4))
    importance.plot(kind="barh")
    plt.title("Feature Importance")
    plt.tight_layout()
    return plt