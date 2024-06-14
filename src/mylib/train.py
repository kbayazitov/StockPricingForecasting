import numpy as np
import torch
from torch import nn
from tqdm.notebook import tqdm

def train_model(model, train_data, test_data, n_steps=4, attempts=2, epochs=30, batch_size=32, lr=1e-3):

    list_of_train_corrs = []
    list_of_test_corrs = []
    list_of_train_losses = []
    list_of_test_losses = []

    for attempt in tqdm(range(attempts)):

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_function = nn.MSELoss()

        train_corrs = []
        test_corrs = []
        train_losses = []
        test_losses = []

        for epoch in tqdm(range(epochs), leave=False):
            train_generator = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
            train_loss = 0

            train_actual_values_per_step = [[] for _ in range(n_steps)]
            train_pred_values_per_step = [[] for _ in range(n_steps)]
            
            model.train()
            for x, y in tqdm(train_generator, leave=False):
                optimizer.zero_grad()
                x = x.to(model.device)
                y = y.to(model.device)
                output = model(x, y)

                loss = loss_function(output, y[:,:,3])

                loss.backward()
                optimizer.step()

                y_true = y[:,:,3].cpu().detach().numpy()
                y_pred = output.cpu().detach().numpy()

                for step in range(n_steps):
                    train_actual_values_per_step[step].extend(y_true[:, step])
                    train_pred_values_per_step[step].extend(y_pred[:, step])
                    
                train_loss += loss.cpu().item()

            test_generator = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
            test_loss = 0

            test_actual_values_per_step = [[] for _ in range(n_steps)]
            test_pred_values_per_step = [[] for _ in range(n_steps)]

            model.eval()
            for x, y in tqdm(test_generator, leave=False):

                x = x.to(model.device)
                y = y.to(model.device)
                output = model(x)
                loss = loss_function(output, y[:,:,3])

                y_true = y[:,:,3].cpu().detach().numpy()
                y_pred = output.cpu().detach().numpy()

                for step in range(n_steps):
                    test_actual_values_per_step[step].extend(y_true[:, step])
                    test_pred_values_per_step[step].extend(y_pred[:, step])
                    
                test_loss += loss.cpu().item()

            train_actual_values_per_step = np.array(train_actual_values_per_step)
            train_pred_values_per_step = np.array(train_pred_values_per_step)
            test_actual_values_per_step = np.array(test_actual_values_per_step)
            test_pred_values_per_step = np.array(test_pred_values_per_step)

            train_corrs.append(
                np.array(
                    [np.corrcoef(train_actual_values_per_step[step], 
                                 train_pred_values_per_step[step])[0, 1] 
                     for step in range(n_steps)]
                ).mean()
            )
            test_corrs.append(
                np.array(
                    [np.corrcoef(test_actual_values_per_step[step], 
                                 test_pred_values_per_step[step])[0, 1] 
                     for step in range(n_steps)]
                ).mean()
            )

            train_losses.append(train_loss*batch_size/len(train_data))
            test_losses.append(test_loss*batch_size/len(test_data))

        list_of_train_corrs.append(train_corrs)
        list_of_test_corrs.append(test_corrs)
        list_of_train_losses.append(train_losses)
        list_of_test_losses.append(test_losses)

    return list_of_test_corrs, list_of_test_losses, list_of_train_corrs, list_of_train_losses
