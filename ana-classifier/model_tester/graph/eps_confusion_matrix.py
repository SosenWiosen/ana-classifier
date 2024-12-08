import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn


def eps_confusion_matrix(y_true,y_pred, savePath):
    cf_matrix = confusion_matrix(y_true, y_pred)

    # Step 3: Normalize the Confusion Matrix
    row_sums = cf_matrix.sum(axis=1)
    normalized_cf_matrix = cf_matrix / row_sums[:, np.newaxis]

    # Step 4: Convert to DataFrame for visualization
    class_names = ['class1', 'class2', 'class3']  # replace with your actual class names
    df_cm = pd.DataFrame(normalized_cf_matrix, index=class_names, columns=class_names)

    # Step 5: Create a Heatmap
    plt.figure(figsize=(10, 7))
    heatmap = sn.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues")
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

    # Optional: Saving the figure to EPS
    plt.savefig(savePath, format='eps', bbox_inches='tight')
