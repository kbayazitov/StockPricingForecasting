import numpy as np
import matplotlib.pyplot as plt

def makeplots(losses, labels, corrs=None, from_n=0, colors=['blue', 'green', 'red', 'orange', 'purple', 'pink', 'black']):
    """
    Draws a plots of losses and accuracies
    Args:
        losses: list - losses
        labels: list - labels of plots
        corrs: list - correlations
        colors: list - colors of plots
        task: ['Classification', 'Regression'] - task name
    Returns:
        plot
    Example:
        >>>
    """
    if (corrs is not None):
        for corr, color, label in zip(corrs, colors, labels):
            corr = [corr[i][from_n:] for i in range(len(corr))]
            mean = np.array(corr).mean(0)
            std = np.array(corr).std(0)
            x_axis = np.arange(from_n, len(mean) + from_n)
            plt.plot(x_axis, mean, color=color, label=label)
            plt.fill_between(x_axis, mean-std, mean+std, alpha=0.3, color=color)
        plt.xlabel('Epochs', fontsize=30)
        plt.ylabel('Corr', fontsize=30)
        plt.legend(loc='best')
        plt.show()

    for loss, color, label in zip(losses, colors, labels):
        loss = [loss[i][from_n:] for i in range(len(loss))]
        mean = np.array(loss).mean(0)
        std = np.array(loss).std(0)
        x_axis = np.arange(from_n, len(mean) + from_n)
        plt.plot(x_axis, mean, color=color, label=label)
        plt.fill_between(x_axis, mean-std, mean+std, alpha=0.3, color=color)
        
    plt.xlabel('Epochs', fontsize=30)
    plt.ylabel('MSE', fontsize=30)
    plt.legend(loc='best')
    plt.show()

def plot_test_results(model, test_dataset, input_size=30, output_size=10, num_rows=5):

    fig, ax = plt.subplots(num_rows, figsize = (13, 15))

    # plot test predictions
    for i in range(num_rows):
        x_input = test_dataset[i * 5][0].unsqueeze(0)
        y_input = test_dataset[i * 5][1].unsqueeze(0)

        pred = model(x_input.to(model.device)).cpu().detach().numpy()
        ax[i].plot(np.linspace(0, input_size, input_size), x_input[0,:,3].tolist(), color='b')
        ax[i].plot(np.linspace(input_size, input_size+output_size+1, output_size+1), [x_input[0,-1,3].item()]+y_input[0,:,3].tolist(), color='g', label='actual')
        ax[i].plot(np.linspace(input_size, input_size+output_size+1, output_size+1), [x_input[0,-1,3].item()]+pred[0].tolist(), color='r', label='pred')
        
    plt.legend(loc='best')
    plt.show()
