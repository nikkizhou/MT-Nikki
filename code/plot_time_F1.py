import matplotlib.pyplot as plt
import numpy as np

# Data
models = [
    "Logistic Regression", "SVM", "Random Forest",
    "BiLSTM", "DistilBERT", "BERT", "Llama"
]
file_name = 'Time_F1_Chart.png'

total_time = [0.42, 1.37, 1.87, 8.28, 30, 58, 523]
#gpu_memory = [0, 0, 0, 119, 534, 868, 9448]
f1_score = [0.73, 0.74, 0.70, 0.61, 0.90, 0.92, 0.91]

x = np.arange(len(models))  # X-axis positions

# Create the figure
fig, axs = plt.subplots(3, 1, figsize=(12, 12))

# Grouped Bar Chart
width = 0.25
axs[0].bar(x - width, total_time, width, label='Total Time (s)', color='blue')
#axs[0].bar(x, gpu_memory, width, label='GPU Memory Used (MB)', color='green')
axs[0].bar(x + width, f1_score, width, label='Weighted F1-Score', color='red')
axs[0].set_title('Grouped Bar Chart: Model Performance and Resource Usage')
axs[0].set_xticks(x)
axs[0].set_xticklabels(models, rotation=45)
axs[0].legend()
axs[0].grid(True, axis='y')

# Dual-Axis Plot
fig2, ax1 = plt.subplots(figsize=(12, 5))

ax2 = ax1.twinx()
ax1.plot(models, total_time, 'bo-', label='Total Time (s)')
#ax1.plot(models, gpu_memory, 'gs-', label='GPU Memory Used (MB)')

ax2.plot(models, f1_score, 'r^-', label='Weighted F1-Score')

ax1.set_xlabel('Model')
ax1.set_ylabel('Time', color='black')
ax2.set_ylabel('Weighted F1-Score', color='red')
fig2.suptitle('Resource Usage vs F1-Score')
ax1.tick_params(axis='x', rotation=45)

# Add legends for both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

# Separate Subplots
axs[1].bar(models, total_time, color='blue')
axs[1].set_title('Total Time (s) per Model')
axs[1].set_xticklabels(models, rotation=45)

#axs[2].bar(models, gpu_memory, color='green')
axs[2].set_title('GPU Memory Used (MB) per Model')
axs[2].set_xticklabels(models, rotation=45)

plt.tight_layout()
# plt.show()
plt.savefig(file_name)