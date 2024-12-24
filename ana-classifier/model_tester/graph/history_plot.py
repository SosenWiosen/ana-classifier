import matplotlib.pyplot as plt

def save_plots_separately(history, save_path, filename='history_plots'):
    metrics = ['accuracy', 'loss', 'f1']
    validation_metrics = ['val_accuracy', 'val_loss', 'val_f1']
    titles = ['Model Accuracy', 'Model Loss', 'Model F1 Score']
    y_labels = ['Accuracy', 'Loss', 'F1 Score']

    for i, metric in enumerate(metrics):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history[metric])
        plt.plot(history.history[validation_metrics[i]])
        plt.title(titles[i])
        plt.ylabel(y_labels[i])
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(f'{save_path}/{filename}_{metric}.eps', format='eps', bbox_inches='tight')
        plt.savefig(f'{save_path}/{filename}_{metric}.svg', format='svg', bbox_inches='tight')
        plt.savefig(f'{save_path}/{filename}_{metric}.png', format='png', bbox_inches='tight')

    plt.close()  # Close the figure to free memory

def save_combined_plot(history, save_dir, filename='combined_plots'):
    fig, axs = plt.subplots(3, 1, figsize=(10, 20))
    metrics = ['accuracy', 'loss', 'f1']
    validation_metrics = ['val_accuracy', 'val_loss', 'val_f1']
    titles = ['Model Accuracy', 'Model Loss', 'Model F1 Score']
    y_labels = ['Accuracy', 'Loss', 'F1 Score']

    for i, metric in enumerate(metrics):
        axs[i].plot(history.history[metric])
        axs[i].plot(history.history[validation_metrics[i]])
        axs[i].set_title(titles[i])
        axs[i].set_ylabel(y_labels[i])
        axs[i].set_xlabel('Epoch')
        axs[i].legend(['Train', 'Val'], loc='upper left')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(f'{save_dir}/{filename}.eps', format='eps', bbox_inches='tight')
    plt.savefig(f'{save_dir}/{filename}.svg', format='svg', bbox_inches='tight')
    plt.savefig(f'{save_dir}/{filename}.png', format='png', bbox_inches='tight')
    plt.close()  # Close the figure to free memory
