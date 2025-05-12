import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the data for each metric
data = {
    'Weighted F1': {
    'title': 'Weighted F1',
    'with_sd': [0.86, 0.91, 0.91, 0.77, 0.72, 0.74, 0.73],
    'without_sd': [0.91, 0.93, 0.90, 0.72, 0.73, 0.74, 0.70],
}

    # 'Accuracy': {
    #     'title': 'Accuracy',
    #     'with_cv': [0.93, 0.96, 0.94, 0.71, 0.73, 0.74, 0.72],
    #     'without_cv': [0.86, 0.91, 0.91, 0.68, 0.73, 0.74, 0.72],
    # },
    # 'Macro Precision': {
    #     'title': 'Macro Precision',
    #     'with_cv': [0.93, 0.95, 0.90, 0.69, 0.74, 0.75, 0.71],
    #     'without_cv': [0.90, 0.89, 0.89, 0.67, 0.79, 0.79, 0.80],
    # },
    # 'Macro Recall': {
    #     'title': 'Macro Recall',
    #     'with_cv': [0.92, 0.96, 0.95, 0.71, 0.73, 0.74, 0.72],
    #     'without_cv': [0.80, 0.90, 0.88, 0.72, 0.67, 0.69, 0.63],
    # },
    # 'Macro F1': {
    #     'title': 'Macro F1',
    #     'with_cv': [0.92, 0.95, 0.94, 0.69, 0.74, 0.75, 0.72],
    #     'without_cv': [0.83, 0.90, 0.90, 0.67, 0.71, 0.73, 0.68],
    # },
    # 'Weighted Precision': {
    #     'title': 'Weighted Precision',
    #     'with_cv': [0.93, 0.96, 0.94, 0.73, 0.74, 0.75, 0.73],
    #     'without_cv': [0.87, 0.91, 0.91, 0.73, 0.75, 0.76, 0.75],
    # },
    # 'Weighted Recall': {
    #     'title': 'Weighted Recall',
    #     'with_cv': [0.93, 0.96, 0.94, 0.72, 0.73, 0.74, 0.72],
    #     'without_cv': [0.86, 0.91, 0.91, 0.68, 0.73, 0.74, 0.72],
    # },
    # 'Weighted F1': {
    #     'title': 'Weighted F1',
    #     'with_cv': [0.93, 0.96, 0.94, 0.72, 0.74, 0.75, 0.73],
    #     'without_cv': [0.86, 0.91, 0.91, 0.69, 0.73, 0.74, 0.70],
    # }
}

# Model names
models = ["Llama", "BERT", "DistilBERT", "Bi-LSTM", "LR", "SVM", "Random Forest"]

# Function to generate and save plots
def generate_and_save_plot(metric_name, metric_data):
    fig, ax = plt.subplots(figsize=(10, 6))

    index = np.arange(len(models))
    bar_width = 0.35

    bar1 = ax.bar(index, metric_data['with_sd'], bar_width, label='With SD', color='#f08080')  
    bar2 = ax.bar(index + bar_width, metric_data['without_sd'], bar_width, label='Without SD', color='#4169e1') 

    # Adding value labels to the bars
    for bar in bar1 + bar2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom')

    # Add labels and title
    ax.set_xlabel('Models')
    ax.set_ylabel(metric_name)
    ax.set_title(metric_data['title'])
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(models)
    ax.legend()

    # Save the plot to a file in the /mnt/data directory
    plt.tight_layout()
    file_name = f'plot/{metric_name.lower().replace(" ", "_")}sd_chart.png'
    plt.savefig(file_name)
    plt.close()

    return file_name

# Generate and save all the plots for each metric
file_paths = [generate_and_save_plot(name, data[name]) for name in data]
file_paths