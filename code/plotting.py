import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the data from the tables
accuracy_data = {
    'Model': ["Llama", "BERT", "DistilBERT", "Bi-LSTM", "LR", "SVM", "Random Forest"],
    'With CV': [0.93, 0.96, 0.94, 0.71, 0.73, 0.74, 0.72],
    'Without CV': [0.86, 0.91, 0.91, 0.68, 0.73, 0.74, 0.72]
}

file_name = 'Macro_Precision_Chart.png'
title = "Macro Precision"


# Convert to DataFrame
df_accuracy = pd.DataFrame(accuracy_data)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Create a bar chart with lighter colors
index = np.arange(len(df_accuracy['Model']))
bar_width = 0.35

bar1 = ax.bar(index, df_accuracy['With CV'], bar_width, label='With CV', color='#f08080')  
bar2 = ax.bar(index + bar_width, df_accuracy['Without CV'], bar_width, label='Without CV', color='#4169e1')  

# Adding value labels to the bars
for bar in bar1 + bar2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords='offset points',
                ha='center', va='bottom')

# Add labels and title
ax.set_xlabel('Models')
ax.set_ylabel(title)
ax.set_title(title)
# ax.set_ylabel('Accuracy')
# ax.set_title('Accuracy')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(df_accuracy['Model'])
ax.legend()

# Save the plot
plt.tight_layout()
plt.savefig(file_name)
# plt.savefig('Accuracy_Chart.png')
plt.close()