# my_model.py

import torch
import torch.nn as nn

class EMGSeq2SeqNet(nn.Module):
    def __init__(self, input_channels, time_len, output_len, 
                 cnn_channels=32, lstm_hidden=128, lstm_layers=1, dropout_rate=0.5):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=(5, 3), padding='same'),
            nn.BatchNorm2d(cnn_channels),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(4, 1)),
            nn.Dropout2d(dropout_rate)
        )

        self.lstm_input_size = cnn_channels * input_channels * (time_len // 4)
        
        self.encoder = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )

        self.decoder = nn.Sequential(
            nn.Linear(lstm_hidden, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(512, output_len)
        )

    def forward(self, x):
        B, S, T, C_in = x.shape
        x = x.view(B * S, T, C_in)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(0, 1, 3, 2)
        x = torch.flatten(x, start_dim=1)
        x = x.view(B, S, -1)
        _, (hidden, _) = self.encoder(x)
        context_vector = hidden[-1]
        output = self.decoder(context_vector)
        return output