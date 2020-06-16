import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        pass
    
    def forward(self, features, captions):
        embedding = self.embed(captions[:,:-1])
        embedding = torch.cat((features.unsqueeze(1), embedding), 1) 
        hiddens, _ = self.lstm(embedding)
        outputs = self.linear(hiddens)
        return outputs
        pass

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sample_indexes = []
        index = 0
        while index < 20:
            hiddens, states = self.lstm(inputs, states) 
            outputs = self.linear(hiddens.squeeze(1))
            predicted_ans = outputs.max(1)[1]
            sample_indexes.append(predicted_ans.tolist()[0])
            inputs = self.embed(predicted_ans)
            inputs = inputs.unsqueeze(1)
            index += 1
            
        return sample_indexes
        pass