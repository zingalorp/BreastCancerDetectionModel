import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def get_misclassified_samples(model, X_test, y_test):
    """
    Returns a DataFrame containing misclassified samples along with their true labels and predictions.
    
    Parameters:
      model: Trained model.
      X_test: Test set features.
      y_test: True labels for the test set.
    """
    # Obtain predictions
    y_pred = model.predict(X_test)
    # Create a DataFrame for easier analysis
    results_df = pd.DataFrame(X_test.copy())
    results_df['true_label'] = y_test.values
    results_df['predicted_label'] = y_pred
    # Filter misclassified examples
    misclassified = results_df[results_df['true_label'] != results_df['predicted_label']]
    return misclassified

def plot_confusion_matrix(model, X_test, y_test, save_path=None):
    """
    Plots and optionally saves the confusion matrix for the given model predictions.
    
    Parameters:
      model: Trained model.
      X_test: Test set features.
      y_test: True labels.
      save_path: Optional file path to save the figure.
    """
    from sklearn.metrics import ConfusionMatrixDisplay
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def print_classification_report(model, X_test, y_test):
    """
    Prints a detailed classification report.
    
    Parameters:
      model: Trained model.
      X_test: Test set features.
      y_test: True labels.
    """
    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
