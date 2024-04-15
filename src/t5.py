from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
import torch

class T5Handler(object):
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small").to("cuda")
        # Alternative T5 variants can be loaded in just by changing these two lines:
        # self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        # self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to("cuda")

        self.output_length = 60
        self.model.min_length = self.output_length
        # Disable gradient calculation for the entire model (freezing the model)
        for param in self.model.parameters():
            param.requires_grad = False

    # Encode a batch of descriptions
    def encode(self, inp):
        descriptions = []
        for sentence in inp:
            descriptions.append(sentence)
        output = self.encodeLanguage(descriptions)[0]

        return output
    
    # Decode a batch of descriptions
    def decode(self, inp):
        output = []
        for i in range(0, inp.shape[0]):
            batchsample = inp[i]
            batchsample = torch.unsqueeze(batchsample, 0)
            out = self.decodeLanguage(batchsample)
            output.append(out)
        import numpy
        output = numpy.asarray(output)
        return output

    # Encode a single description
    def encodeLanguage(self, descriptions):
        input_ids = self.tokenize(descriptions)
        encoder_outputs = self.model.encoder(input_ids=input_ids.to("cuda"))
        return encoder_outputs

    # Decode a single description
    def decodeLanguage(self, encoder_outputs):
        decoder_input_ids = [self.model.config.decoder_start_token_id]
        predicted_ids = []

        # manually decode tokens inside of the limited output range
        for i in range(self.output_length):
            outputs = self.model.decoder(input_ids=torch.tensor([decoder_input_ids]).to("cuda"), encoder_hidden_states=encoder_outputs.to("cuda"))
            logits = self.model.lm_head(outputs[0])
            logits = logits[:,i,:]
            predicted_id = logits.argmax(-1) # perform argmax on the last dimension (i.e. greedy decoding)
            if predicted_id.item() == 1 :
                break
            predicted_ids.append(predicted_id.item()) # add predicted id to decoder_input_ids
            decoder_input_ids = decoder_input_ids + [predicted_id]

        return self.tokenizer.decode(predicted_ids)

    # Decode a batch of descriptions
    def decodeEntireBatch(self, batch, inp, targets):
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
        decoder_input_ids = self.model._shift_right(targets)

        decoder_outputs = self.model.decoder(input_ids=decoder_input_ids.to("cuda"), encoder_hidden_states=inp.to("cuda"))
        logits = self.model.lm_head(decoder_outputs[0])

        batch_loss = loss_func(logits.view(-1, logits.size(-1)).to("cuda"), targets.view(-1).to("cuda"))

        return batch_loss

    # Tokenizes the given text with the T5 Tokenizer
    def tokenize(self, text):
        return self.tokenizer(text, padding=True, return_tensors="pt").input_ids

    # Detokenizes the given tokens with the T5 Tokenizer
    def detokenize(self, tokens):
        return self.tokenizer.decode(tokens)
    
    # Pad by a specified amount
    def pad(self, encoding, amount):
        batch = encoding.shape[0]
        padding = amount - encoding.shape[1]
        dimension = encoding.shape[2]
        return torch.cat((encoding, torch.zeros(batch, padding, dimension).to("cuda")), dim=1)