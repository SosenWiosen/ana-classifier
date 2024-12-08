import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn


def export_confusion_matrix(y_true, y_pred, class_names, savePath):
    cf_matrix = confusion_matrix(y_true, y_pred)

    # Step 3: Normalize the Confusion Matrix
    row_sums = cf_matrix.sum(axis=1)
    normalized_cf_matrix = cf_matrix / row_sums[:, np.newaxis]

    df_cm = pd.DataFrame(normalized_cf_matrix, index=class_names, columns=class_names)

    # Step 5: Create a Heatmap
    plt.figure(figsize=(10, 7))
    heatmap = sn.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues")
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

    # Optional: Saving the figure to EPS
    plt.savefig(f'{savePath}/cfm.eps', format='eps', bbox_inches='tight')
    plt.savefig(f'{savePath}/cfm.png', format='png', bbox_inches='tight')
    plt.savefig(f'{savePath}/cfm.svg', format='svg', bbox_inches='tight')


