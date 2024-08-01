class TrackerAutoEncoder(nn.Module):
    def __init__(self, tracker_size, num_condition_frames, hidden_size, latent_size):
        super().__init__()
        self.input_size = tracker_size * num_condition_frames
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size + self.input_size, self.latent_size)

    def encode(self, tracker_data):
        data = tracker_data.flatten(-2)
        out = F.elu(self.fc1(data))
        out = F.elu(self.fc2(torch.cat((out, data), dim=1)))
        out = F.elu(self.fc3(torch.cat((out, data), dim=1)))
        out = self.fc4((torch.cat((out, data), dim=1)))
        return out

    def forward(self, tracker_data):
        latent = self.encode(tracker_data)
        return latent

class TrackerAutoEncoderV2(nn.Module):
    def __init__(self, tracker_size, num_condition_frames, hidden_size, latent_size):
        super().__init__()
        self.input_size = tracker_size * num_condition_frames
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc5 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc6 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc7 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc8 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc9 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc10 = nn.Linear(self.hidden_size + self.input_size, self.latent_size)

    def encode(self, tracker_data):
        data = tracker_data.flatten(-2)
        out = F.elu(self.fc1(data))
        out = F.elu(self.fc2(torch.cat((out, data), dim=1)))
        out = F.elu(self.fc3(torch.cat((out, data), dim=1)))
        out = F.elu(self.fc4(torch.cat((out,data)),dim=1))
        out = F.elu(self.fc5(torch.cat((out, data)), dim=1))
        out = F.elu(self.fc6(torch.cat((out, data)), dim=1))
        out = F.elu(self.fc7(torch.cat((out, data)), dim=1))
        out = F.elu(self.fc8(torch.cat((out, data)), dim=1))
        out = F.elu(self.fc9(torch.cat((out, data)), dim=1))
        out = self.fc10((torch.cat((out, data), dim=1)))
        return out

    def forward(self, tracker_data):
        latent = self.encode(tracker_data)
        return latent
class TrackerAutoDecoder(nn.Module):
    def __init__(self, latent_size, tracker_size, num_condition_frames, hidden_size, output_size):
        super().__init__()
        self.input_size = latent_size
        self.tracker_size = tracker_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size + self.input_size, self.output_size)

    def forward(self, latent, tracker):
        out = F.elu(self.fc1(latent))
        out = F.elu(self.fc2(torch.cat((out, latent), dim=1)))
        out = F.elu(self.fc3(torch.cat((out, latent), dim=1)))
        out = self.fc4(torch.cat((out, latent), dim=1))
        return out

class TrackerAutoDecoderV2(nn.Module):
    def __init__(self, latent_size, tracker_size, num_condition_frames, hidden_size, output_size):
        super().__init__()
        self.input_size = latent_size
        self.tracker_size = tracker_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc5 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc6 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc7 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc8 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc9 = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.fc10 = nn.Linear(self.hidden_size + self.input_size, self.output_size)

    def forward(self, latent, tracker):
        out = F.elu(self.fc1(latent))
        out = F.elu(self.fc2(torch.cat((out, latent), dim=1)))
        out = F.elu(self.fc3(torch.cat((out, latent), dim=1)))
        out = F.elu(self.fc4(torch.cat((out, latent), dim=1)))
        out = F.elu(self.fc5(torch.cat((out, latent), dim=1)))
        out = F.elu(self.fc6(torch.cat((out, latent), dim=1)))
        out = F.elu(self.fc7(torch.cat((out, latent), dim=1)))
        out = F.elu(self.fc8(torch.cat((out, latent), dim=1)))
        out = F.elu(self.fc9(torch.cat((out, latent), dim=1)))
        out = self.fc10(torch.cat((out, latent), dim=1))
        return out

# paramter like Linear 같은것 맟추기....
class TrackerAutoV2(nn.Module):
    def __init__(self, tracker_size, num_condition_frames, encoder_hidden_size, latent_size, decoder_hidden_size,
                 output_size):
        super().__init__()
        self.encoder = TrackerAutoEncoderV2(tracker_size, num_condition_frames, encoder_hidden_size, latent_size)

        self.decoder = TrackerAutoDecoderV2(latent_size, tracker_size, num_condition_frames, decoder_hidden_size,
                                          output_size)
        self.fc1=nn.Linear(tracker_size,tracker_size)
        self.fc2 = nn.Linear(tracker_size, tracker_size)
        self.fc3 = nn.Linear(tracker_size, tracker_size)
        self.fc4 = nn.Linear(tracker_size, tracker_size)
        self.fc5 = nn.Linear(tracker_size, tracker_size)
        self.fc6 = nn.Linear(tracker_size, tracker_size)
        self.fc7 = nn.Linear(tracker_size, tracker_size)
        self.fc8 = nn.Linear(tracker_size, tracker_size)
        self.fc9 = nn.Linear(tracker_size, tracker_size)
        self.fc10 = nn.Linear(tracker_size, tracker_size)
        self.num_condition_frames = num_condition_frames
        # self.decoder = MixedDecoder(35,latent_size,decoder_hidden_size,0,1,2)

    def forward(self, tracker_data):
        out=F.elu(self.fc1(tracker_data))
        out = F.elu(self.fc2(out))
        out = F.elu(self.fc3(out))
        out = F.elu(self.fc4(out))
        out = F.elu(self.fc5(out))
        out = F.elu(self.fc6(out))
        out = F.elu(self.fc7(out))
        out = F.elu(self.fc8(out))
        out = F.elu(self.fc9(out))
        out = F.elu(self.fc10(out))
        z = self.encoder(out)
        return self.decoder(z, tracker_data[:, int(self.num_condition_frames / 2 - 1), :])
