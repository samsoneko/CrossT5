import torch
from torch import nn
import math


def train(model, t5, batch, optimiser, epoch_loss, params, appriori_len = True, loss_mode="mixed"):
    optimiser.zero_grad()  # free the optimiser from previous gradients

    # set the targets for the training
    mode_sig = torch.randint(10, (1,)) # determines how much time is spent on the translate signal
    if mode_sig == 0: # translate signal
        signal = 'translate'
        if loss_mode == 't5_decoder':
            gt_language = t5.tokenize(list(batch['T_t_fw']))
        elif loss_mode == 'h_vector' or loss_mode == 'mixed':
            gt_language = t5.encode(['Translate English to German: ' + d for d in list(batch['T_fw'])]) # take translate input as target
        gt_action = batch['B_fw'][0].repeat(len(batch['B_fw'][1:]), 1, 1) * batch["B_bin"][1:] # take initial joint configuration repeated as target
    else: # ptae signal
        ptae_sig = torch.randint(0, 3, (1,))
        if ptae_sig == 0: # repeat signal
            repeat_sig = torch.randint(2, (1,))
            if repeat_sig == 0: # repeat action signal
                signal = 'repeat action'
                if loss_mode == 'h_vector':
                    gt_prepad = t5.encode(['' for d in list(batch['L_fw'])]) # take empty encoding as target
                    gt_language = t5.pad(gt_prepad, 20)
                elif loss_mode == 't5_decoder' or loss_mode == 'mixed':
                    gt_language = t5.tokenize(['' for d in list(batch['L_fw'])])
                gt_action = batch['B_fw'][1:] # take all datapoints except the first as a target
            else: # repeat language signal
                signal = 'repeat language'
                if loss_mode == 'h_vector':
                    gt_prepad = t5.encode(['Translate English to English: ' + d for d in list(batch['L_fw'])]) # take encodings that output the original string as target
                    gt_language = t5.pad(gt_prepad, 20)
                elif loss_mode == 't5_decoder' or loss_mode == 'mixed':
                    gt_language = t5.tokenize(list(batch['L_fw']))
                gt_action = batch['B_fw'][0].repeat(len(batch['B_fw'][1:]), 1, 1) * batch["B_bin"][1:] # take initial joint configuration repeated as target
        elif ptae_sig == 1: # describe signal
            signal = 'describe'
            if loss_mode == 'h_vector':
                gt_prepad = t5.encode(['Translate English to English: ' + d for d in list(batch['L_fw'])]) # take encodings that output the original string as target
                gt_language = t5.pad(gt_prepad, 20)
            elif loss_mode == 't5_decoder' or loss_mode == 'mixed':
                gt_language = t5.tokenize(list(batch['L_fw']))
            gt_action = batch['B_bw'][0].repeat(len(batch['B_fw'][1:]), 1, 1) * batch["B_bin"][1:] # take last joint configuration repeated as target
        else: # execute signal
            signal = 'execute'
            if loss_mode == 'h_vector':
                gt_prepad = t5.encode(['' for d in list(batch['L_fw'])]) # take empty encoding as target
                gt_language = t5.pad(gt_prepad, 20)
            elif loss_mode == 't5_decoder' or loss_mode == 'mixed':
                gt_language = t5.tokenize(['' for d in list(batch['L_fw'])])
            gt_action = batch['B_fw'][1:] # take all datapoints except the first as a target

    if signal != 'translate' and loss_mode == 'h_vector':
        tnum = gt_prepad.size(1)
    else:
        tnum = 0

    output = model(batch, signal) # run the model with the data
    L_loss, B_loss, batch_loss = loss(batch, tnum, output, gt_language, gt_action, batch["B_bin"], signal, params, loss_mode, t5)  # compute loss
    batch_loss.backward()  # compute gradients
    optimiser.step()  # update weights
    epoch_loss.append(batch_loss.item()) # record the batch loss
    # scheduler.step()

    return L_loss, B_loss, batch_loss, signal  # return the losses

def validate(model, t5, batch, epoch_loss, params, appriori_len = True, loss_mode="mixed"):
    with torch.no_grad(): # run without calculating gradients
        mode_sig = torch.randint(10, (1,)) # determines how much time is spent on the translate signal
        if mode_sig == 0: # translate signal
            signal = 'translate'
            if loss_mode == 't5_decoder':
                gt_language = t5.tokenize(list(batch['T_t_fw'])) # take translate target as target
            elif loss_mode == 'h_vector' or loss_mode == 'mixed':
                gt_language = t5.encode(['Translate English to German: ' + d for d in list(batch['T_fw'])]) # take encoded translate input as target
            gt_action = batch['B_fw'][0].repeat(len(batch['B_fw'][1:]), 1, 1) * batch["B_bin"][1:] # take initial joint configuration repeated as target
        else: # ptae signal
            ptae_sig = torch.randint(0, 3, (1,))
            if ptae_sig == 0: # repeat signal
                repeat_sig = torch.randint(2, (1,))
                if repeat_sig == 0: # repeat action signal
                    signal = 'repeat action'
                    if loss_mode == 'h_vector':
                        gt_prepad = t5.encode(['' for d in list(batch['L_fw'])]) # take empty encoding as target
                        gt_language = t5.pad(gt_prepad, 20)
                    elif loss_mode == 't5_decoder' or loss_mode == 'mixed':
                        gt_language = t5.tokenize(['' for d in list(batch['L_fw'])]) # take empty encoding as target
                    gt_action = batch['B_fw'][1:] # take all datapoints except the first as a target
                else: # repeat language signal
                    signal = 'repeat language'
                    if loss_mode == 'h_vector':
                        gt_prepad = t5.encode(['Translate English to English: ' + d for d in list(batch['L_fw'])]) # take encodings that output the original string as target
                        gt_language = t5.pad(gt_prepad, 20)
                    elif loss_mode == 't5_decoder' or loss_mode == 'mixed':
                        gt_language = t5.tokenize(list(batch['L_fw'])) # take language input as target
                    gt_action = batch['B_fw'][0].repeat(len(batch['B_fw'][1:]), 1, 1) * batch["B_bin"][1:] # take initial joint configuration repeated as target
            elif ptae_sig == 1: # describe signal
                signal = 'describe'
                if loss_mode == 'h_vector':
                    gt_prepad = t5.encode(['Translate English to English: ' + d for d in list(batch['L_fw'])]) # take encodings that output the original string as target
                    gt_language = t5.pad(gt_prepad, 20)
                elif loss_mode == 't5_decoder' or loss_mode == 'mixed':
                    gt_language = t5.tokenize(list(batch['L_fw'])) # take language input as target
                gt_action = batch['B_bw'][0].repeat(len(batch['B_fw'][1:]), 1, 1) * batch["B_bin"][1:] # take last joint configuration repeated as target
            else: # execute signal
                signal = 'execute'
                if loss_mode == 'h_vector':
                    gt_prepad = t5.encode(['' for d in list(batch['L_fw'])]) # take empty encoding as target
                    gt_language = t5.pad(gt_prepad, 20)
                elif loss_mode == 't5_decoder' or loss_mode == 'mixed':
                    gt_language = t5.tokenize(['' for d in list(batch['L_fw'])]) # take empty encoding as target
                gt_action = batch['B_fw'][1:] # take all datapoints except the first as a target
        
        if signal != 'translate' and loss_mode == 'h_vector':
            tnum = gt_prepad.size(1)
        else:
            tnum = 0

        output = model(batch, signal) # run the model with the data
        L_loss, B_loss, batch_loss = loss(batch, tnum, output, gt_language, gt_action, batch["B_bin"], signal, params, loss_mode, t5)  # compute loss
        epoch_loss.append(batch_loss.item())  # record the batch loss

    return L_loss, B_loss, batch_loss, signal # return the losses

def loss(batch, tnum, output, gt_language, gt_action, B_bin, signal, params, loss_mode, t5):
    [L_output, B_output] = output

    if loss_mode == 't5_decoder':
        L_loss = t5.decodeEntireBatch(batch, L_output, gt_language)
    elif loss_mode == 'h_vector':
        if signal == 'translate':
            L_loss = torch.mean(torch.square(L_output - gt_language)) # language loss (MSE)
        else:
            L_loss = torch.mean(torch.square(L_output[:,:tnum,:] - gt_language[:,:tnum,:])) # language loss (MSE)
    elif loss_mode == 'mixed':
        if signal == 'translate':
            L_loss = torch.mean(torch.square(L_output - gt_language)) # language loss (MSE)
        else:
            L_loss = t5.decodeEntireBatch(batch, L_output, gt_language)

    B_output = B_output * B_bin[1:]
    B_loss = torch.mean(torch.square(B_output.to("cuda") - gt_action.to("cuda"))) # action loss (MSE)

    loss = params.L_weight * L_loss + params.B_weight * B_loss
    return L_loss, B_loss, loss

# Word Embedding Layer
class Embedder(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
    def forward(self, x):
        return self.embed(x)

class PeepholeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, peephole=False, forget_bias=0.0):
        super().__init__()
        self.input_sz = input_size
        self.hidden_size = hidden_size
        self.peephole = peephole
        self.W = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size * 4))
        self.peep_i = nn.Parameter(torch.Tensor(hidden_size))
        self.peep_f = nn.Parameter(torch.Tensor(hidden_size))
        self.peep_o = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.forget_bias = forget_bias
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, sequence_len=None,
                init_states=None):
        """Assumes x is of shape (sequence, batch, feature)"""
        if sequence_len is None:
            seq_sz, bs, _ = x.size()
        else:
            seq_sz = sequence_len.max()
            _, bs, _ = x.size()
        hidden_seq = []
        if init_states is None:
            c_t, h_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            c_t, h_t = init_states

        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[t, :, :]
            if sequence_len is not None:
                if sequence_len.min() <= t+1:
                    old_c_t = c_t.clone().detach()
                    old_h_t = h_t.clone().detach()
            # batch the computations into a single matrix multiplication
            lstm_mat = torch.cat([x_t, h_t], dim=1)
            if self.peephole:
                gates = lstm_mat @ self.W + self.bias
            else:
                gates = lstm_mat @ self.W + self.bias
                g_t = torch.tanh(gates[:, HS * 2:HS * 3])

            if self.peephole:
                i_t, j_t, f_t, o_t = (
                    (gates[:, :HS]),  # input
                    (gates[:, HS:HS * 2]),  # new input
                    (gates[:, HS * 2:HS * 3]),   # forget
                    (gates[:, HS * 3:])   # output
                )
            else:
                i_t, f_t, o_t = (
                    torch.sigmoid(gates[:, :HS]),  # input
                    torch.sigmoid(gates[:, HS:HS * 2]),# + self.forget_bias),  # forget
                    torch.sigmoid(gates[:, HS * 3:])  # output
                )

            if self.peephole:
                c_t = torch.sigmoid(f_t + self.forget_bias + c_t * self.peep_f) * c_t \
                      + torch.sigmoid(i_t + c_t * self.peep_i) * torch.tanh(j_t)
                h_t = torch.sigmoid(o_t + c_t * self.peep_o) * torch.tanh(c_t)
            else:
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)

            out = h_t.clone()
            if sequence_len is not None:
                if sequence_len.min() <= t:
                    c_t = torch.where(torch.tensor(sequence_len).to(c_t.device) <= t, old_c_t.T, c_t.T).T
                    h_t = torch.where(torch.tensor(sequence_len).to(h_t.device) <= t, old_h_t.T, h_t.T).T
                    out = torch.where(torch.tensor(sequence_len).to(out.device) <= t, torch.zeros(out.shape).to(out.device).T, out.T).T

            hidden_seq.append(out.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)

        return hidden_seq, (h_t, c_t)
   
class Encoder(nn.Module):
    def __init__(self, params, lstm_type='peephole'):
        super(Encoder, self).__init__()
        self.params = params
        self.lstm_type = lstm_type
        self.enc_cells = torch.nn.Sequential()
        if self.lstm_type == 'bidirectional':
            self.enc_cells.add_module("ealstm", nn.LSTM(input_size=self.params.VB_input_dim,
                                                                    hidden_size=self.params.VB_num_units,
                                                                    num_layers=self.params.VB_num_layers,
                                                                    bidirectional=True))
        elif self.lstm_type == 'peephole':
            for i in range(self.params.VB_num_layers):
                if i == 0:
                    self.enc_cells.add_module("ealstm" + str(i), PeepholeLSTM(input_size=self.params.VB_input_dim,
                                                                                    hidden_size=self.params.VB_num_units,
                                                                                    peephole=True, forget_bias=0.8))
                else:
                    self.enc_cells.add_module("ealstm" + str(i), PeepholeLSTM(input_size=self.params.VB_num_units,
                                                                                    hidden_size=self.params.VB_num_units,
                                                                                    peephole=True, forget_bias=0.8))
        else:
            self.enc_cells.add_module("ealstm", nn.LSTM(input_size=self.params.VB_input_dim,
                                                                    hidden_size=self.params.VB_num_units,
                                                                    num_layers=self.params.VB_num_layers,
                                                                    bidirectional=False))
    def forward(self, inp, sequence_length):
        num_of_layers = self.params.VB_num_layers
        layer_input = inp
        if self.lstm_type == 'peephole':
            for l in range(num_of_layers):
                enc_cell = self.enc_cells.__getitem__(l)
                hidden_seq, _ = enc_cell(layer_input.float().to('cuda'), sequence_len=sequence_length)
                layer_input = hidden_seq
        else:
            enc_cell = self.enc_cells.__getitem__(0)
            hidden_seq, _ = enc_cell(layer_input.float().to('cuda'))
            
        return hidden_seq.permute(1,0,2)

class Decoder(nn.Module):
    def __init__(self, params, lstm_type='peephole', appriori_len=True):
        super(Decoder, self).__init__()
        self.params = params
        self.lstm_type = lstm_type
        self.dec_cells = torch.nn.Sequential()
        for i in range(self.params.VB_num_layers):
            if i == 0:
                if self.lstm_type == 'peephole':
                    self.dec_cells.add_module("dalstm"+str(i), PeepholeLSTM(input_size=self.params.VB_input_dim,
                                                                            hidden_size=self.params.VB_num_units,
                                                                            peephole=True, forget_bias=0.8).to('cuda'))
                else:
                    self.dec_cells.add_module("dalstm"+str(i), nn.LSTM(input_size=self.params.VB_input_dim,
                                                                            hidden_size=self.params.VB_num_units).to('cuda'))
            else:
                if self.lstm_type == 'peephole':
                    self.dec_cells.add_module("dalstm"+str(i), PeepholeLSTM(input_size=self.params.VB_num_units,
                                                                            hidden_size=self.params.VB_num_units,
                                                                            peephole=True, forget_bias=0.8).to('cuda'))
                else:
                    self.dec_cells.add_module("dalstm"+str(i), nn.LSTM(input_size=self.params.VB_num_units,
                                                                            hidden_size=self.params.VB_num_units).to('cuda'))
        if appriori_len:
            self.linear = nn.Linear(self.params.VB_num_units, self.params.B_input_dim)
        else:
            self.sigmoid = nn.Sigmoid()
            self.linear_pred = nn.Linear(self.params.VB_num_units, self.params.B_input_dim-1)
            self.linear_finish = nn.Linear(self.params.VB_num_units, 1)
        self.tanh = nn.Tanh()

    def forward(self, input, length=None, initial_state=None, teacher_forcing=True):
        y = []

        self.vis_out = False

        initial_state = initial_state.view(initial_state.size()[0], self.params.VB_num_layers, 2, self.params.VB_num_units)
        initial_state = initial_state.permute(1, 2, 0, 3)
        if length == None:
            done = torch.zeros(input[1].shape[0], dtype=bool).cuda()
            for i in range(self.params.B_max_length - 1):
                if self.vis_out == False or teacher_forcing == True:
                    if len(input[0]) > i:
                        current_V_in = input[0][i]
                    else:
                        current_V_in = torch.zeros((input[0].shape[1], input[0].shape[2])).cuda()#input[0][len(input[0])-1]
                dec_states = []
                if i == 0:
                    if self.vis_out and teacher_forcing == False:
                        current_V_in = input[0]
                    current_B_in = input[-1]#input
                    layer_input = torch.cat([current_V_in, current_B_in], dim=1).unsqueeze(0)#current_B_in.unsqueeze(0)#
                    for j in range(self.params.VB_num_layers):
                        dec_state = (initial_state[j][0].float(), initial_state[j][1].float())
                        dec_cell = self.dec_cells.__getitem__(j)
                        if self.lstm_type == 'peephole':
                            output, (hx, cx) = dec_cell(layer_input.float(), init_states=dec_state)
                        else:
                            output, (hx, cx) = dec_cell(layer_input.float().to('cuda'), (dec_state[0].unsqueeze(0).contiguous(),
                                                                                            dec_state[1].unsqueeze(0).contiguous()))
                        dec_state = (hx, cx)
                        dec_states.append(dec_state)
                        layer_input = output
                else:
                    if self.vis_out and teacher_forcing == False:
                        layer_input = out
                    else:
                        if self.vis_out and teacher_forcing == True:
                            current_B_in = out[:,:,30:].squeeze(dim=0)
                        else:
                            current_B_in = out.squeeze(dim=0)
                        layer_input = torch.cat([current_V_in, current_B_in], dim=1).unsqueeze(0)#current_B_in.unsqueeze(0)#
                    for j in range(self.params.VB_num_layers):
                        dec_cell = self.dec_cells.__getitem__(j)
                        dec_state = prev_dec_states[j]
                        if self.lstm_type == 'peephole':
                            output, (hx,cx) = dec_cell(layer_input.float(), init_states=dec_state)
                        else:
                            output, (hx, cx) = dec_cell(layer_input.float(), dec_state)
                        dec_state = (hx, cx)
                        dec_states.append(dec_state)
                        layer_input = output
                prev_dec_states = dec_states
                linear = self.linear_pred(layer_input)
                out_dim = self.tanh(linear)
                finish = self.linear_finish(layer_input)
                out_finish = self.sigmoid(finish)
                out = torch.cat((out_dim, out_finish), -1)#torch.cat((out_dim.squeeze(0), out_finish.squeeze(0)), -1)
                y.append(out.squeeze(0))  #y.append(out)
                for batch_ind in (done == False).nonzero(as_tuple=True)[0]:
                    if out_finish.squeeze(0)[batch_ind] < 0.5:
                        done[batch_ind] = True
                if torch.all(done):
                    break
        else:
            for i in range(length - 1):
                if self.vis_out == False or teacher_forcing == True:
                    current_V_in = input[0][i]
                dec_states = []
                if i == 0:
                    if self.vis_out and teacher_forcing == False:
                        current_V_in = input[0]
                    current_B_in = input[-1]#input
                    layer_input = torch.cat([current_V_in, current_B_in], dim=1).unsqueeze(0)#current_B_in.unsqueeze(0)#
                    for j in range(self.params.VB_num_layers):
                        dec_state = (initial_state[j][0].float(), initial_state[j][1].float())
                        dec_cell = self.dec_cells.__getitem__(j)
                        if self.lstm_type == 'peephole':
                            output, (hx, cx) = dec_cell(layer_input.float(), init_states=dec_state)
                        else:
                            output, (hx, cx) = dec_cell(layer_input.float().to('cuda'), (dec_state[0].unsqueeze(0).contiguous(),
                                                                                            dec_state[1].unsqueeze(0).contiguous()))
                        dec_state = (hx, cx)
                        dec_states.append(dec_state)
                        layer_input = output
                else:
                    if self.vis_out and teacher_forcing == False:
                        layer_input = out
                    else:
                        if self.vis_out and teacher_forcing == True:
                            current_B_in = out[:,:,30:].squeeze(dim=0)
                        else:
                            current_B_in = out.squeeze(dim=0)
                        layer_input = torch.cat([current_V_in, current_B_in], dim=1).unsqueeze(0)#current_B_in.unsqueeze(0)#
                    for j in range(self.params.VB_num_layers):
                        dec_cell = self.dec_cells.__getitem__(j)
                        dec_state = prev_dec_states[j]
                        if self.lstm_type == 'peephole':
                            output, (hx,cx) = dec_cell(layer_input.float(), init_states=dec_state)
                        else:
                            output, (hx, cx) = dec_cell(layer_input.float(), dec_state)
                        dec_state = (hx, cx)
                        dec_states.append(dec_state)
                        layer_input = output
                prev_dec_states = dec_states
                linear = self.linear(layer_input)
                out = self.tanh(linear)
                y.append(out.squeeze())
        y = torch.stack(y, dim=0)
        return y

class PTAE(nn.Module):
    def __init__(self, t5, params, lang_enc_type='T5', act_enc_type='LSTM', appriori_len=True, addnorm_lang=False):
        super(PTAE, self).__init__()
        from crossmodal_transformer import Visual_Ling_Attn as CMTransformer
        self.params = params
        self.lang_enc_type = lang_enc_type
        self.act_enc_type = act_enc_type
        self.addnorm_lang = addnorm_lang

        self.lang_encoder = t5
        self.action_encoder = Encoder(self.params, lstm_type='bidirectional')

        self.hidden = CMTransformer(self.params)

        self.initial_lang = nn.Linear(self.params.hidden_dim, self.params.L_num_units*self.params.L_num_layers)
        self.initial_act = nn.Linear(self.params.hidden_dim, self.params.VB_num_units*self.params.VB_num_layers*2)

        self.action_decoder = Decoder(self.params, lstm_type='regular', appriori_len=True)

    def forward(self, inp, signal):
        # Signal addition and action transform
        if signal == 'translate':
            lang_inp = ['Translate English to German: ' + d for d in list(inp['T_fw'])] # add translate language signal to input
            B_input = inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)
            V_input = torch.zeros(len(inp['V_fw']), inp['V_fw'].shape[:2][1], 30).cuda()
            VB_input = torch.cat((V_input, B_input), dim=2)
        elif signal == 'repeat language':
            lang_inp = ['repeat language: ' + d for d in list(inp['L_fw'])] # add repeat language signal to input
            VB_forward = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2)
            VB_bin = inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            VB_input =  VB_forward * VB_bin
        elif signal == 'execute':
            lang_inp = ['execute: ' + d for d in list(inp['L_fw'])] # add execute signal to input
            VB_forward = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2)
            VB_bin = inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            VB_input =  VB_forward * VB_bin
        elif signal == 'repeat action':
            lang_inp = ['repeat action:' for d in list(inp['L_fw'])] # add repeat action signal to input
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
        else:
            lang_inp = ['describe:' for d in list(inp['L_fw'])] # add describe signal to input
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)

        # Encoding the language
        if self.lang_enc_type == 'T5':
            encoded_lang = self.lang_encoder.encode(lang_inp)
        else:
            encoded_lang = lang_inp.permute(1,0,2).float()

        # If in nico mode, padding language with zeros to length of 30
        if signal != 'translate':
            encoded_lang = self.lang_encoder.pad(encoded_lang, 20)

        # Encoding the action
        if self.act_enc_type == 'LSTM':
            encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())
        else:
            encoded_act = VB_input.permute(1,0,2).float()

        # Running the crossmodal transformer
        full_h = self.hidden(encoded_lang, encoded_act, None, None)
        mean_h = full_h.mean(1)

        # Using full h as language output
        L_dec_init_state = self.initial_lang(full_h)
        L_output = torch.add(encoded_lang, L_dec_init_state)

        # Decoding the action with mean h
        VB_dec_init_state = self.initial_act(mean_h)
        if signal == 'describe':
            VB_input_f = [inp["V_bw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_bw"][0, :, :]]
            B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
        elif signal == 'repeat language':
            VB_input_f = [inp["V_fw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_fw"][0, :, :]]
            B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
        else:
            VB_input_f = inp['VB_fw']
            B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)

        return L_output, B_output

    def inference(self, inp, signal, appriori_len=True):
        # Signal addition and action transform
        if signal == 'translate':
            lang_inp = ['Translate English to German: ' + d for d in list(inp['T_fw'])] # add repeat language signal to input
            B_input = inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)
            V_input = torch.zeros(len(inp['V_fw']), inp['V_fw'].shape[:2][1], 30).cuda()
            VB_input = torch.cat((V_input, B_input), dim=2)
        elif signal == 'repeat language':
            lang_inp = ['repeat language: ' + d for d in list(inp['L_fw'])] # add repeat language signal to input
            VB_forward = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2)
            VB_bin = inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            VB_input =  VB_forward * VB_bin
        elif signal == 'execute':
            lang_inp = ['execute: ' + d for d in list(inp['L_fw'])] # add execute signal to input
            VB_forward = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2)
            VB_bin = inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            VB_input =  VB_forward * VB_bin
        elif signal == 'repeat action':
            lang_inp = ['repeat action:' for d in list(inp['L_fw'])] # add repeat action signal to input
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
        else:
            lang_inp = ['describe:' for d in list(inp['L_fw'])] # add describe signal to input
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)

        # Encoding the language
        if self.lang_enc_type == 'T5':
            encoded_lang = self.lang_encoder.encode(lang_inp)
        else:
            encoded_lang = lang_inp.permute(1,0,2).float()

        # If in nico mode, padding language with zeros to length of 30
        if signal != 'translate':
            encoded_lang = self.lang_encoder.pad(encoded_lang, 20)

        # Encoding the action
        if self.act_enc_type == 'LSTM':
            encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())
        else:
            encoded_act = VB_input.permute(1,0,2).float()

        # Running the crossmodal transformer
        full_h = self.hidden(encoded_lang, encoded_act, None, None)
        mean_h = full_h.mean(1)

        # Using full h as language output
        L_dec_init_state = self.initial_lang(full_h)
        L_output = torch.add(encoded_lang, L_dec_init_state)

        # Decoding the action with mean h
        VB_dec_init_state = self.initial_act(mean_h)
        if signal == 'describe':
            VB_input_f = [inp["V_bw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_bw"][0, :, :]]
            B_output = self.action_decoder(VB_input_f, inp['B_len'], VB_dec_init_state)
        elif signal == 'repeat language':
            VB_input_f = [inp["V_fw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_fw"][0, :, :]]
            B_output = self.action_decoder(VB_input_f, inp['B_len'], VB_dec_init_state)
        else:
            VB_input_f = inp['VB_fw']
            B_output = self.action_decoder(VB_input_f, inp['B_len'], VB_dec_init_state)

        return L_output, B_output

    def extract_representations(self, inp, signal):
        # Signal addition and action transform
        if signal == 'translate':
            lang_inp = ['Translate English to German: ' + d for d in list(inp['T_fw'])] # add repeat language signal to input
            B_input = inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)
            V_input = torch.zeros(len(inp['V_fw']), inp['V_fw'].shape[:2][1], 30).cuda()
            VB_input = torch.cat((V_input, B_input), dim=2)
        elif signal == 'repeat language':
            lang_inp = ['repeat language: ' + d for d in list(inp['L_fw'])] # add repeat language signal to input
            VB_forward = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2)
            VB_bin = inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            VB_input =  VB_forward * VB_bin
        elif signal == 'execute':
            lang_inp = ['execute: ' + d for d in list(inp['L_fw'])] # add execute signal to input
            VB_forward = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2)
            VB_bin = inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            VB_input =  VB_forward * VB_bin
        elif signal == 'repeat action':
            lang_inp = ['repeat action:' for d in list(inp['L_fw'])] # add repeat action signal to input
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
        else:
            lang_inp = ['describe:' for d in list(inp['L_fw'])] # add describe signal to input
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)

        # Encoding the language
        if self.lang_enc_type == 'T5':
            encoded_lang = self.lang_encoder.encode(lang_inp)
        else:
            encoded_lang = lang_inp.permute(1,0,2).float()

        # If in nico mode, padding language with zeros to length of 30
        if signal != 'translate':
            encoded_lang = self.lang_encoder.pad(encoded_lang, 20)

        # Encoding the action
        if self.act_enc_type == 'LSTM':
            encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())
        else:
            encoded_act = VB_input.permute(1,0,2).float()

        # Running the crossmodal transformer
        full_h = self.hidden(encoded_lang, encoded_act, None, None)

        return full_h