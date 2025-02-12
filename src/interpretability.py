import matplotlib.pyplot as plt
import pandas as pd

def plot_logistic_coefficients(model, feature_names, top_n=10, save_path=None):
    """
    Displays a bar chart of the top features based on the absolute value of the coefficients
    from a trained logistic regression model.
    
    Parameters:
      model: Trained LogisticRegression model.
      feature_names: List or array of feature names.
      top_n: Number of top features to display.
    """
    # Extract coefficients (assuming a binary classification with shape (1, n_features))
    coefficients = model.coef_[0]
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    })
    # Order features by the magnitude of the coefficient
    coef_df['abs_coefficient'] = coef_df['coefficient'].abs()
    coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(coef_df['feature'].head(top_n), coef_df['coefficient'].head(top_n))
    plt.gca().invert_yaxis()
    plt.xlabel("Coefficient Value")
    plt.title("Top Feature Coefficients in Logistic Regression")
    plt.tight_layout()
    if save_path:
            plt.savefig(save_path)
    plt.show()


# SHAP analysis
def plot_shap_summary(model, background_data, test_data, feature_names, save_path=None):
    """
    Generates a SHAP summary plot and returns SHAP values.
    """
    import shap
    import matplotlib.pyplot as plt

    bg_data = background_data.values if hasattr(background_data, "values") else background_data
    test_data_np = test_data.values if hasattr(test_data, "values") else test_data

    explainer = shap.Explainer(model, bg_data)
    shap_values = explainer(test_data_np)
    
    plt.figure()
    shap.summary_plot(shap_values, test_data_np, feature_names=feature_names, show=False)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    
    return shap_values

def plot_shap_waterfall(shap_values, test_data, feature_names, instance_index=0, save_path=None):
    """
    Generates a SHAP waterfall plot for a specific test instance.
    """
    import shap
    import matplotlib.pyplot as plt
    
    instance_data = test_data.iloc[instance_index] if hasattr(test_data, 'iloc') else test_data[instance_index]   

    plt.figure()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[instance_index].values,
            base_values=shap_values[instance_index].base_values,
            data=instance_data,
            feature_names=feature_names
        ),
        show=False
    )
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()



