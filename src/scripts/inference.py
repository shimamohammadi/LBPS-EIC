import torch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from scripts import customDataset
from scripts import model_design
import pandas as pd
import numpy as np


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def inference_with_dropout(model, x):
    """ Perform inference with dropout enabled """
    model.eval()  # Set the model to evaluation mode
    enable_dropout(model)  # Enable dropout during inference
    return model(x)


def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_design.Architecture(1, rate=0.2).to(device)
    checkpoint = torch.load('./src/model/LBPS_0.2Dropout.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    enable_dropout(model)
    return model


def load_data(ref_dir, dis_dir, all_image_triplets_names):

    # Test set
    test_dataset = customDataset.CustomDataset(x_path=all_image_triplets_names,
                                               ref_dir=ref_dir,
                                               dis_dir=dis_dir,
                                               transform=customDataset.custom_transform)

    test_loader = DataLoader(dataset=test_dataset,
                             drop_last=True,
                             shuffle=False,
                             num_workers=1)

    return test_loader


def get_inference(ref_dir, dis_dir, all_image_triplets_names):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model()

    batch_size = 1

    test_loader = load_data(ref_dir, dis_dir, all_image_triplets_names)

    data_preference_lst = []

    loop_test = enumerate(test_loader, 1)
    with torch.no_grad():
        for batch_idx, (refs, dis_1s, dis_2s) in loop_test:

            refs = refs.to(device)
            dis_1s = dis_1s.to(device)
            dis_2s = dis_2s.to(device)

            data_preference = model(refs, dis_1s, dis_2s)

            data_preference_lst.append(data_preference.detach().item())

    return data_preference_lst


def MC_drop_out(all_image_triplets_names, ref_dir, dis_dir, num_pairs):
    MC_ITERATIONS = 500

    data_pr_arr = np.zeros((MC_ITERATIONS, num_pairs))

    # For every model
    for mc_iter in range(0, MC_ITERATIONS):

        # Get inference
        data_preference_lst = get_inference(
            ref_dir, dis_dir, all_image_triplets_names)

        data_pr_arr[mc_iter] = data_preference_lst

        print(mc_iter)

    data_pr_df = pd.DataFrame(data_pr_arr)

    monte_carlo_arr = np.zeros((num_pairs, 2))

    for i in range(num_pairs):

        monte_carlo_arr[i, 0] = data_pr_df.iloc[:, i].mean()
        monte_carlo_arr[i, 1] = data_pr_df.iloc[:, i].var()

    monte_carlo_df = pd.DataFrame(monte_carlo_arr, columns=[
                                  'Model_preference', 'Model_Uncertainty'])
    return monte_carlo_df
