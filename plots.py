import matplotlib.pyplot as plt


def plot_comparisons(y_true, predictions, metrics):
    '''
    Function to plot the comparison between the real values and the predicted values

    INPUT:
        y_true (pd.Series): Real values
        predictions (dict): Dictionary containing the predictions of each model
        metrics (dict): Dictionary containing the R^2 score of each model

    OUTPUT:
        None
    '''

    plt.figure(figsize=(18, 10))
    
    for i, (model_name, y_pred) in enumerate(predictions.items(), 1):
        plt.subplot(1, 1, i)
        plt.plot(range(len(y_true)), y_true, label='Real', color='blue')
        plt.plot(range(len(y_pred)), y_pred, label='Predicted', linestyle='dashed', color='red')
        plt.xlabel('Samples')
        plt.ylabel('Knee angle on the x axis (Deg)')
        plt.legend()
        plt.title(f'{model_name} - R^2 Score: {metrics[model_name]:.2f}')
    
    plt.tight_layout()
    plt.show()