import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from datetime import datetime

def plot_metrics(learner, save_dir="plots"):
    if learner.metrics.n == 0:
        print("No metrics to plot yet.")
        return

    # create directory if not exists
    os.makedirs(save_dir, exist_ok=True)

    # generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"metrics_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)

    # create plot
    plt.figure(figsize=(10, 6))
    plt.plot(learner.metrics.history_n, learner.metrics.history_mae, label="MAE")
    plt.plot(learner.metrics.history_n, learner.metrics.history_rmse, label="RMSE")
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xlabel("Number of Samples Seen")
    plt.ylabel("Error")
    plt.title("Model Evaluation Metrics Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # save AND show
    plt.savefig(filepath, dpi=200)
    plt.show()

    print(f"📊 Metrics plot saved to: {filepath}")
