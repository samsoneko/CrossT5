import torch
from ptae import PTAE
from config import PtaeConfig, TrainConfig
from data_util import save_latent, save_text
import numpy as np
from dataset import CLANT
from torch.utils.data import DataLoader
from data_util import normalise, pad_with_zeros, add_active_feature
from proprioception_eval_robustness import evaluate
from nltk.translate.bleu_score import sentence_bleu
from t5 import T5Handler

# Find the descriptions via given actions
def inference(signal, checkpoint_name, robustness_mode):
    # get the network configuration (parameters such as number of layers and units)
    ptae_conf = PtaeConfig()
    ptae_conf.set_conf("../train/ptae_conf.txt")

    # get the training configuration
    # (batch size, initialisation, number of iterations, saving and loading directory)
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")
    save_dir = train_conf.save_dir
    appriori_len = train_conf.appriori_len

    # configuration settings that only apply to the inference
    write_to_file = True # determines if the inference results should be written to a file, or just printed on the console
    write_directory = '../train/evaluation/' # determines the writing directory
    variants = True # determines if vocabulary variants should be included in testing

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    print("The currently selected GPU is number:", torch.cuda.current_device(), ", it's a ", torch.cuda.get_device_name(device=None),)
    # Create a T5 instance
    t5 = T5Handler()
    # Create a model instance
    model = PTAE(t5, ptae_conf, appriori_len=appriori_len).to(device)
    # Load the trained model
    checkpoint = torch.load(save_dir + checkpoint_name)       # get the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])       # load the model state

    # Load the dataset
    training_data = CLANT(train_conf, remhyph=True)
    test_data = CLANT(train_conf, True, remhyph=True)

    # Get the max and min values for normalisation
    max_joint = np.concatenate((training_data.B_fw, test_data.B_fw), 1).max()
    min_joint = np.concatenate((training_data.B_fw, test_data.B_fw), 1).min()
    max_vis = np.concatenate((training_data.V_fw, test_data.V_fw), 1).max()
    min_vis = np.concatenate((training_data.V_fw, test_data.V_fw), 1).min()

    # normalise the joint angles, visual features between -1 and 1 and pad the fast actions with zeros
    training_data.B_bw = normalise(training_data.B_bw, max_joint, min_joint) * training_data.B_bin
    training_data.B_fw = normalise(training_data.B_fw, max_joint, min_joint) * training_data.B_bin
    test_data.B_bw = normalise(test_data.B_bw, max_joint, min_joint) * test_data.B_bin
    test_data.B_fw = normalise(test_data.B_fw, max_joint, min_joint) * test_data.B_bin
    training_data.V_bw = normalise(training_data.V_bw, max_vis, min_vis) * training_data.V_bin
    training_data.V_fw = normalise(training_data.V_fw, max_vis, min_vis) * training_data.V_bin
    test_data.V_bw = normalise(test_data.V_bw, max_vis, min_vis) * test_data.V_bin
    test_data.V_fw = normalise(test_data.V_fw, max_vis, min_vis) * test_data.V_bin

    # Load the training and testing sets with DataLoader
    train_dataloader = DataLoader(training_data)
    test_dataloader = DataLoader(test_data)

    model.eval()
    file = open('../CLANT/vocabularytc.txt', 'r')
    vocab = file.read().splitlines()
    train_true = 0
    train_false = 0
    test_true = 0
    test_false = 0

    if signal == 'describe' or signal == 'repeat language' or signal == 'translate':
        train_bleu_score = 0
        test_bleu_score = 0

    # Feed the dataset as input
    for input in train_dataloader:
        if variants:
            sentence_idx = np.random.randint(4)  # Generate random index for description alternatives
        else:
            sentence_idx = 0
        # Get the target language output
        if signal == 'translate':
            L_fw_before = input["T_t_fw"]
        else:
            L_fw_before = input["L_fw"]
            L_fw_before = [var[0] for var in L_fw_before]
        # Transpose (batch, datapoints, features) -> (datapoints, batch, features)
        input["B_fw"] = input["B_fw"].transpose(0, 1).to(device)
        input["V_fw"] = input["V_fw"].transpose(0, 1).to(device)
        input["B_bw"] = input["B_bw"].transpose(0, 1).to(device)
        input["V_bw"] = input["V_bw"].transpose(0, 1).to(device)
        input["B_bin"] = input["B_bin"].transpose(0, 1).to(device)
        input["VB_fw"] = [input["V_fw"][:, :, :], input["B_fw"][0, :, :]]
        # Choose one of the four description alternatives according to the generated random index
        # Language Format: (alternatives, batch)
        input["L_fw"] = input["L_fw"][sentence_idx]
        input["T_fw"] = input["T_fw"][0]

        if robustness_mode == 0:
            input["L_fw"] = input["L_fw"]
        elif robustness_mode == 1:
            input["L_fw"] = ('please ' + input["L_fw"][0],)
        elif robustness_mode == 2:
            input["L_fw"] = ('would you please ' + input["L_fw"][0],)
        elif robustness_mode == 3:
            input["L_fw"] = (input["L_fw"][0].rsplit(' ', 1)[0] + ' the ' + input["L_fw"][0].rsplit(' ', 1)[-1] + ' cube',)
        elif robustness_mode == 4:
            input["L_fw"] = (input["L_fw"][0].rsplit(' ', 1)[0] + ' the ' + input["L_fw"][0].rsplit(' ', 1)[-1] + ' cube now',)
        elif robustness_mode == 5:
            input["L_fw"] = ('please ' + input["L_fw"][0].rsplit(' ', 1)[0] + ' the ' + input["L_fw"][0].rsplit(' ', 1)[-1] + ' cube',)
        elif robustness_mode == 6:
            input["L_fw"] = ('would you please ' + input["L_fw"][0].rsplit(' ', 1)[0] + ' the ' + input["L_fw"][0].rsplit(' ', 1)[-1] + ' cube',)
        elif robustness_mode == 7:
            input["L_fw"] = (input["L_fw"][0].rsplit(' ', 1)[-1] + ' ' + input["L_fw"][0].rsplit(' ', 1)[0],)
        elif robustness_mode == 8:
            input["L_fw"] = ('the ' + input["L_fw"][0].rsplit(' ', 1)[-1] + ' cube ' + input["L_fw"][0].rsplit(' ', 1)[0],)
        elif robustness_mode == 9:
            command = input["L_fw"][0]
            command = command.replace("push", "shove")
            command = command.replace("pull", "drag")
            input["L_fw"] = (command,)

        # Run the model
        with torch.no_grad():
            lang_result, act_result = model.inference(input, signal, appriori_len)
        
        # Decode language with the T5 Decoder and save it to file
        lang_result = t5.decode(lang_result)
        save_text(lang_result, input["L_filenames"][0], "inference/" + checkpoint_name.lstrip('/').rstrip('.tar') + "_robustness/" + str(robustness_mode) + "/" + signal + "/") # save the predicted descriptions

        # Format and save action to file
        act_result = act_result.cpu()
        act_result = (((act_result+1)/2)*(input["max_joint"]-input["min_joint"]))+input["min_joint"] # get back raw values
        save_latent(act_result.unsqueeze(0), input["B_filenames"][0], "inference/" + checkpoint_name.lstrip('/').rstrip('.tar') + "_robustness/" + str(robustness_mode) + "/" + signal + "/")
        
        print(input["B_filenames"][0])

        r = lang_result[0] # language result of the current model run

        if signal == 'translate':
            enc = t5.encode(['Translate English to German: ' + input["T_fw"][0]])
            dec = t5.decode(enc)
            t = dec[0]
        elif signal == 'describe':
            t = L_fw_before # target for the current model run
        else:
            t = [L_fw_before[sentence_idx]]

        # cumulative BLEU scores
        # Check if predicted descriptions match the original ones
        if signal == 'translate':
            target = [[]]
            prediction = []
            if r == t:
                print(True)
                train_true = train_true + 1
                print(r)
                train_bleu_score = train_bleu_score + 1.0
            else:
                print(False)
                train_false = train_false + 1
                print("Input: " + str(input["T_fw"]))
                print("Expected by original T5: " + str(t))
                print("Respective dataset target: " + str([L_fw_before[0][0]]))
                print("Produced: " + str(r))
                target[0] = (t.split())
                prediction = (r.split())
                train_bleu_score = train_bleu_score + sentence_bleu(target, prediction, weights=(1/2, 1/2))
            print("BLEU Score: " + str(train_bleu_score))
        if signal == 'describe' or signal == 'repeat language':
            target = [[]]
            prediction = []
            if r in t:
                print(True)
                train_true = train_true + 1
                print(r)
                train_bleu_score = train_bleu_score + 1.0
            else:
                print(False)
                train_false = train_false + 1
                print("Expected: " + str(t))
                target[0] = (t[0].split())
                prediction = (r.split())
                print("Produced: " + str(r))
                train_bleu_score = train_bleu_score + sentence_bleu(target, prediction, weights=(1/2, 1/2))
            print("BLEU Score: " + str(train_bleu_score))
        elif signal == 'execute' or signal == 'repeat action':
            if r == "":
                print(True)
                train_true = train_true + 1
                print(r)
            else:
                print(False)
                train_false = train_false + 1
                print("Expected: ")
                print("Produced: " + str(r))
        print()
    training_accuracy = train_true / (train_true + train_false)
    if signal == 'describe' or signal == 'repeat language' or signal == 'translate':
        training_bleu2 = train_bleu_score / (train_true + train_false)

    # Do the same for the test set
    if train_conf.test:
        print("test!")
        for input in test_dataloader:
            if variants:
                sentence_idx = np.random.randint(4)  # Generate random index for description alternatives
            else:
                sentence_idx = 0
            # Get the target language output
            if signal == 'translate':
                L_fw_before = input["T_t_fw"]
            else:
                L_fw_before = input["L_fw"]
                L_fw_before = [var[0] for var in L_fw_before]
            # Transpose (batch, datapoints, features) -> (datapoints, batch, features)
            input["B_fw"] = input["B_fw"].transpose(0, 1).to(device)
            input["V_fw"] = input["V_fw"].transpose(0, 1).to(device)
            input["B_bw"] = input["B_bw"].transpose(0, 1).to(device)
            input["V_bw"] = input["V_bw"].transpose(0, 1).to(device)
            input["B_bin"] = input["B_bin"].transpose(0, 1).to(device)
            input["VB_fw"] = [input["V_fw"][:, :, :], input["B_fw"][0, :, :]]
            # Choose one of the four description alternatives according to the generated random index
            # Language Format: (alternatives, batch)
            input["L_fw"] = input["L_fw"][sentence_idx]
            input["T_fw"] = input["T_fw"][0]
            
            if robustness_mode == 0:
                input["L_fw"] = input["L_fw"]
            elif robustness_mode == 1:
                input["L_fw"] = ('please ' + input["L_fw"][0],)
            elif robustness_mode == 2:
                input["L_fw"] = ('would you please ' + input["L_fw"][0],)
            elif robustness_mode == 3:
                input["L_fw"] = (input["L_fw"][0].rsplit(' ', 1)[0] + ' the ' + input["L_fw"][0].rsplit(' ', 1)[-1] + ' cube',)
            elif robustness_mode == 4:
                input["L_fw"] = (input["L_fw"][0].rsplit(' ', 1)[0] + ' the ' + input["L_fw"][0].rsplit(' ', 1)[-1] + ' cube now',)
            elif robustness_mode == 5:
                input["L_fw"] = ('please ' + input["L_fw"][0].rsplit(' ', 1)[0] + ' the ' + input["L_fw"][0].rsplit(' ', 1)[-1] + ' cube',)
            elif robustness_mode == 6:
                input["L_fw"] = ('would you please ' + input["L_fw"][0].rsplit(' ', 1)[0] + ' the ' + input["L_fw"][0].rsplit(' ', 1)[-1] + ' cube',)
            elif robustness_mode == 7:
                input["L_fw"] = (input["L_fw"][0].rsplit(' ', 1)[-1] + ' ' + input["L_fw"][0].rsplit(' ', 1)[0],)
            elif robustness_mode == 8:
                input["L_fw"] = ('the ' + input["L_fw"][0].rsplit(' ', 1)[-1] + ' cube ' + input["L_fw"][0].rsplit(' ', 1)[0],)
            elif robustness_mode == 9:
                command = input["L_fw"][0]
                command = command.replace("push", "shove")
                command = command.replace("pull", "drag")
                input["L_fw"] = (command,)

            # Run the model
            with torch.no_grad():
                lang_result, act_result = model.inference(input, signal, appriori_len)
            
            # Decode language with the T5 Decoder and save it to file
            lang_result = t5.decode(lang_result)
            save_text(lang_result, input["L_filenames"][0], "inference/" + checkpoint_name.lstrip('/').rstrip('.tar') + "_robustness/" + str(robustness_mode) + "/" + signal + "/") # save the predicted descriptions

            # Format and save action to file
            act_result = act_result.cpu()
            act_result = (((act_result+1)/2)*(input["max_joint"]-input["min_joint"]))+input["min_joint"] # get back raw values
            save_latent(act_result.unsqueeze(0), input["B_filenames"][0], "inference/" + checkpoint_name.lstrip('/').rstrip('.tar') + "_robustness/" + str(robustness_mode) + "/" + signal + "/")

            r = lang_result[0] # result of the current model run

            if signal == 'translate':
                enc = t5.encode(['Translate English to German: ' + input["T_fw"][0]])
                dec = t5.decode(enc)
                t = dec[0]
            elif signal == 'describe':
                t = L_fw_before # target for the current model run
            else:
                t = [L_fw_before[sentence_idx]]

            # cumulative BLEU scores
            # Check if predicted descriptions match the original ones
            if signal == 'translate':
                target = [[]]
                prediction = []
                if r == t:
                    print(True)
                    test_true = test_true + 1
                    print(r)
                    test_bleu_score = test_bleu_score + 1
                else:
                    print(False)
                    test_false = test_false + 1
                    print("Input: " + str(input["T_fw"]))
                    print("Expected by original T5: " + str(t))
                    print("Respective dataset target: " + str([L_fw_before[0][0]]))
                    print("Produced: " + str(r))
                    target[0] = (t.split())
                    prediction = (r.split())
                    test_bleu_score = test_bleu_score + sentence_bleu(target, prediction, weights=(1/2, 1/2))
                print("BLEU Score: " + str(test_bleu_score))
            if signal == 'describe' or signal == 'repeat language':
                target = [[]]
                prediction = []
                if r in t:
                    print(True)
                    test_true = test_true + 1
                    print(r)
                    test_bleu_score = test_bleu_score + 1
                else:
                    print(False)
                    test_false = test_false + 1
                    print("Expected: " + str(t))
                    target[0] = (t[0].split())
                    prediction = (r.split())
                    print("Produced: " + str(r))
                    test_bleu_score = test_bleu_score + sentence_bleu(target, prediction, weights=(1/2, 1/2))
                print("BLEU Score: " + str(test_bleu_score))
            elif signal == 'execute' or signal == 'repeat action':
                if r == "":
                    print(True)
                    test_true = test_true + 1
                    print(r)
                else:
                    print(False)
                    test_false = test_false + 1
                    print("Expected: ")
                    print("Produced: " + str(r))
            print()
        test_accuracy = test_true / (test_true + test_false)
        if signal == 'describe' or signal == 'repeat language' or signal == 'translate':
            test_bleu2 = test_bleu_score / (test_true + test_false)

    nrmse_train, nrmse_test = evaluate(signal, appriori_len, checkpoint_name, robustness_mode)

    print('Training sentence accuracy:', "{0:.2%}".format(training_accuracy))
    if signal == 'describe' or signal == 'repeat language' or signal == 'translate':
        print('Training BLUE-2 Score:', training_bleu2)
    print('Normalised Root-Mean squared error (NRMSE) for predicted train joint values, in percent: ', nrmse_train)

    if train_conf.test:
        print('Test sentence accuracy:', "{0:.2%}".format(test_accuracy))
        if signal == 'describe' or signal == 'repeat language' or signal == 'translate':
            print('Test BLUE-2 Score:', test_bleu2)
        print('Normalised Root-Mean squared error (NRMSE) for predicted test joint values, in percent: ', nrmse_test)

    if write_to_file:
        file_name = checkpoint_name.lstrip('/').rstrip('.tar')
        write_file = open(write_directory + "robustness/" + file_name + "_" + str(robustness_mode) + ".txt", 'a')
        write_file.write(signal + ':\n')
        write_file.write('Training sentence accuracy: ' + "{0:.2%}".format(training_accuracy) + '\n')
        if signal == 'describe' or signal == 'repeat language' or signal == 'translate':
            write_file.write('Training BLEU-2 Score: ' + "{0:.2%}".format(training_bleu2) + '\n')
        write_file.write('Normalised Root-Mean squared error (NRMSE) for predicted train joint values, in percent: ' + str(nrmse_train) + '\n')
        if train_conf.test:
            write_file.write('Test sentence accuracy: ' + "{0:.2%}".format(test_accuracy) + '\n')
            if signal == 'describe' or signal == 'repeat language' or signal == 'translate':
                write_file.write('Test BLEU-2 Score: ' + "{0:.2%}".format(test_bleu2) + '\n')
            write_file.write('Normalised Root-Mean squared error (NRMSE) for predicted test joint values, in percent: ' + str(nrmse_test) + '\n')
        write_file.write('\n')
        write_file.close()

def main(checkpoint):
    inference('execute', checkpoint, 0)
    inference('execute', checkpoint, 1)
    inference('execute', checkpoint, 2)
    inference('execute', checkpoint, 3)
    inference('execute', checkpoint, 4)
    inference('execute', checkpoint, 5)
    inference('execute', checkpoint, 6)
    inference('execute', checkpoint, 7)
    inference('execute', checkpoint, 8)
    inference('execute', checkpoint, 9)

if __name__ == "__main__":
    main('/cross_t5.tar')