import torch
import torch.nn as nn

class CSPSeq2SeqNet(nn.Module):
    def __init__(self, csp_components=4, seq_len=5, time_len=500, output_len=1000,
                 cnn_channels=32, lstm_hidden=128, lstm_layers=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool2d((4, 1))  # 500 → 125
        )

        self.lstm_input_size = cnn_channels * csp_components * (time_len // 4)
        self.encoder = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.decoder = nn.Sequential(
            nn.Linear(seq_len * lstm_hidden, 512),
            nn.ReLU(),
            nn.Linear(512, output_len)
        )

    def forward(self, x):
        B, S, T, C = x.shape
        x = x.reshape(B * S, 1, T, C)
        x = self.conv(x)              # (B*S, ch, 125, C)
        x = x.reshape(B, S, -1)       # (B, S, features)
        lstm_out, _ = self.encoder(x) # (B, S, hidden)
        x = lstm_out.reshape(B, -1)   # (B, S*hidden)
        return self.decoder(x)        # (B, 1000)
    
class RawSeq2SeqNet(nn.Module):
    def __init__(self, input_channels=19, seq_len=5, time_len=500, output_len=1000,
                 cnn_channels=32, lstm_hidden=128, lstm_layers=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool2d((4, 1))  # 500 → 125
        )

        self.lstm_input_size = cnn_channels * input_channels * (time_len // 4)
        self.encoder = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.decoder = nn.Sequential(
            nn.Linear(seq_len * lstm_hidden, 512),
            nn.ReLU(),
            nn.Linear(512, output_len)
        )

    def forward(self, x):
        B, S, T, C = x.shape
        x = x.reshape(B * S, 1, T, C)
        x = self.conv(x)              # (B*S, ch, 125, C)
        x = x.reshape(B, S, -1)       # (B, S, features)
        lstm_out, _ = self.encoder(x) # (B, S, hidden)
        x = lstm_out.reshape(B, -1)   # (B, S*hidden)
        return self.decoder(x)        # (B, 1000)