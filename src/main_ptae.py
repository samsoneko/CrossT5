import datetime
import math
import os

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import PtaeConfig, TrainConfig
from data_util import add_active_feature, normalise, pad_with_zeros
from dataset import CLANT
from ptae import PTAE, train, validate
from t5 import T5Handler


def main():
    # get the network configuration (parameters such as number of layers and units)
    ptae_conf = PtaeConfig()
    ptae_conf.set_conf("../train/ptae_conf.txt")

    # get the training configuration
    # (batch size, initialisation, num_of_iterations number, saving and loading directory)
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")
    seed = train_conf.seed
    batch_size = train_conf.batch_size
    num_of_epochs = train_conf.num_of_epochs
    learning_rate = train_conf.learning_rate
    save_dir = train_conf.save_dir
    appriori_len = train_conf.appriori_len # determines whether the model expects input of identical length or not
    loss_mode = train_conf.loss_mode # determines the loss mode used by the model, can be either "mixed", "h_vector" or "t5_decoder"
    if not os.path.exists(os.path.dirname(save_dir)):
        os.mkdir(os.path.dirname(save_dir))

    # Random Initialisation
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Use GPU if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    print("The currently selected GPU is number:", torch.cuda.current_device(), ", it's a ", torch.cuda.get_device_name(device=None),)
    # Create a T5 instance
    t5 = T5Handler()
    # Create a model instance
    model = PTAE(t5, ptae_conf, appriori_len=appriori_len).to(device)
    # Initialise the optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimiser
    scheduler = MultiStepLR(optimiser, milestones=[10000], gamma=0.5)

    #  Inspect the model with tensorboard
    model_name = "cross_t5"
    date = str(datetime.datetime.now()).split(".")[0]
    writer = SummaryWriter(log_dir=".././logs/" + model_name + date)  # initialize the writer with folder "./logs"

    # Load the trained model
    # checkpoint = torch.load(save_dir + '/ptae.tar')       # get the checkpoint
    # model.load_state_dict(checkpoint['model_state_dict'])       # load the model state
    # optimiser.load_state_dict(checkpoint['optimiser_state_dict'])   # load the optimiser state

    model.train()  # tell the model that it's training time

    # Load the dataset
    training_data = CLANT(train_conf, remhyph=True)
    test_data = CLANT(train_conf, True, remhyph=True)

    # Get the max and min values for normalisation (B, V)
    max_joint = np.concatenate((training_data.B_fw, test_data.B_fw), 1).max()
    min_joint = np.concatenate((training_data.B_fw, test_data.B_fw), 1).min()
    max_vis = np.concatenate((training_data.V_fw, test_data.V_fw), 1).max()
    min_vis = np.concatenate((training_data.V_fw, test_data.V_fw), 1).min()

    # normalise the joint angles, visual features between -1 and 1 and pad the fast actions with zeros (B, V)
    training_data.B_bw = normalise(training_data.B_bw, max_joint, min_joint) * training_data.B_bin
    training_data.B_fw = normalise(training_data.B_fw, max_joint, min_joint) * training_data.B_bin
    test_data.B_bw = normalise(test_data.B_bw, max_joint, min_joint) * test_data.B_bin
    test_data.B_fw = normalise(test_data.B_fw, max_joint, min_joint) * test_data.B_bin
    training_data.V_bw = normalise(training_data.V_bw, max_vis, min_vis) * training_data.V_bin
    training_data.V_fw = normalise(training_data.V_fw, max_vis, min_vis) * training_data.V_bin
    test_data.V_bw = normalise(test_data.V_bw, max_vis, min_vis) * test_data.V_bin
    test_data.V_fw = normalise(test_data.V_fw, max_vis, min_vis) * test_data.V_bin

    # Load the training and testing sets with DataLoader
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    step = 0

    # Training
    for epoch in range(num_of_epochs):
        print("")
        print('\033[94m' + "Start of Epoch {}".format(epoch) + '\033[0m')
        epoch_loss = []
        epoch_loss_describe = []
        epoch_loss_execute = []
        print('\033[92m' + "Training Phase" + '\033[0m')
        for input in train_dataloader:
            # Transpose (batch, datapoints, features) -> (datapoints, batch, features)
            input["B_fw"] = input["B_fw"].transpose(0, 1).to(device)
            input["V_fw"] = input["V_fw"].transpose(0, 1).to(device)
            input["B_bw"] = input["B_bw"].transpose(0, 1).to(device)
            input["V_bw"] = input["V_bw"].transpose(0, 1).to(device)
            input["B_bin"] = input["B_bin"].transpose(0, 1).to(device)
            input["VB_fw"] = [input["V_fw"][:, :, :], input["B_fw"][0, :, :]]
            sentence_idx = np.random.randint(4)  # Generate random index for description alternatives
            # Choose one of the four description alternatives according to the generated random index
            # Language Format: (alternatives, batch)
            input["L_fw"] = input["L_fw"][sentence_idx]
            input["T_fw"] = input["T_fw"][0]
            input["T_t_fw"] = input["T_t_fw"][0]

            # Train and print the losses
            l, b, t, signal = train(model, t5, input, optimiser, epoch_loss, ptae_conf, appriori_len, loss_mode)
            # Print info about the current step
            print("step:{} total:{}, language:{}, behavior:{}, signal:{}".format(step, t, l, b, signal))

            step = step + 1
            if signal == "describe":
                epoch_loss_describe.append(epoch_loss[-1])
            elif signal == "execute":
                epoch_loss_execute.append(epoch_loss[-1])

        writer.add_scalar("Training Loss", np.mean(epoch_loss), epoch) # add the overall loss to the Tensorboard
        writer.add_scalar("Training Loss - Describe", np.mean(epoch_loss_describe), epoch) # add the describe loss to the Tensorboard
        writer.add_scalar("Training Loss - Execute", np.mean(epoch_loss_execute), epoch) # add the execute loss to the Tensorboard
        scheduler.step()
        # scheduler.step(np.mean(epoch_loss))
        # Testing
        if train_conf.test and (epoch + 1) % train_conf.test_interval == 0:
            epoch_loss_t = []
            epoch_loss_t_describe = []
            epoch_loss_t_execute = []
            print('\033[93m' + "Testing Phase" + '\033[0m')
            for input in test_dataloader:
                input["B_fw"] = input["B_fw"].transpose(0, 1).to(device)
                input["V_fw"] = input["V_fw"].transpose(0, 1).to(device)
                input["B_bw"] = input["B_bw"].transpose(0, 1).to(device)
                input["V_bw"] = input["V_bw"].transpose(0, 1).to(device)
                input["B_bin"] = input["B_bin"].transpose(0, 1).to(device)
                input["VB_fw"] = [input["V_fw"][:, :, :], input["B_fw"][0, :, :]]
                sentence_idx = np.random.randint(4)  # Generate random index for description alternatives
                # Choose one of the four description alternatives according to the generated random index
                # Language Format: (alternatives, batch)
                input["L_fw"] = input["L_fw"][sentence_idx]
                input["T_fw"] = input["T_fw"][0]
                input["T_t_fw"] = input["T_t_fw"][0]

                # Calculate and print the losses
                l, b, t, signal = validate(model, t5, input, epoch_loss_t, ptae_conf, loss_mode=loss_mode)
                print("step:{} total:{}, language:{}, behavior:{}, signal:{}".format(step, t, l, b, signal))

                if signal == "describe":
                    epoch_loss_t_describe.append(epoch_loss_t[-1])
                elif signal == "execute":
                    epoch_loss_t_execute.append(epoch_loss_t[-1])
                    
            writer.add_scalar("Test Loss", np.mean(epoch_loss_t), epoch) # add the overall loss to the Tensorboard
            writer.add_scalar("Test Loss - Describe", np.mean(epoch_loss_t_describe), epoch) # add the describe loss to the Tensorboard
            writer.add_scalar("Test Loss - Execute", np.mean(epoch_loss_t_execute), epoch) # add the execute loss to the Tensorboard

        # Save the model parameters at every log interval
        if (epoch + 1) % train_conf.log_interval == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                },
                save_dir + "/cross_t5.tar",
            )

    # Flush and close the summary writer of Tensorboard
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
