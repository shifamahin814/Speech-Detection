import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(model_dim, num_heads),
            num_layers
        )
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        transformed = self.transformer_layers(embedded)
        output = self.fc_out(transformed)
        return F.log_softmax(output, dim=-1)

# Example usage:
# model = TransformerModel(input_dim=30, model_dim=64, num_heads=4, num_layers=3, output_dim=5)
# x = torch.rand(10, 32, 30)  # (sequence_length, batch_size, input_dim)
# output = model(x)  # predictions for 5 classes