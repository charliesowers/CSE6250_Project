import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np

class RNN(nn.Module):
	def __init__(self, dim_input):
		super(RNN, self).__init__()
		# baseline model
		self.fc1 = nn.Linear(in_features=dim_input, out_features=100)
		self.rnn = nn.GRU(input_size=100, hidden_size=200, num_layers=2, batch_first=True, bidirectional=True, dropout=0.25)
		self.fc2 = nn.Linear(in_features=200*2, out_features=2)
		self.dropout = nn.Dropout(0.25)

	def forward(self, input_tuple):

		seqs, lengths = input_tuple
		seqs = torch.relu(self.fc1(seqs))
		seqs = self.dropout(seqs)
		packed_input = pack_padded_sequence(seqs, lengths, batch_first=True)
		rnn_output, _ = self.rnn(packed_input)
		padded_output, _ = pad_packed_sequence(rnn_output, batch_first=True)
		padded_output = padded_output[np.arange(len(padded_output)), lengths-1]
		out = self.fc2(padded_output)
		return out


# Define the content model, a hybrid TopicRNN model
class Content(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes, embedding_dim, num_topic, bidirectional=False):
		super(Content, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.num_classes = num_classes
		self.bidirectional = bidirectional
		self.embedding_dim = embedding_dim
		self.num_topic = num_topic
		self.fc1 = nn.Linear(in_features=input_size, out_features=embedding_dim)
		self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
						  bidirectional=self.bidirectional, batch_first=True, dropout=0.25)
		self.rn = RecognitionNetwork(input_size, hidden_size, num_topic)
		self.fc2 = nn.Linear(in_features=hidden_size * num_layers, out_features=1)
		self.dropout = nn.Dropout(0.25)

	def forward(self, input_tuple):
		seqs, lengths = input_tuple

		# calculate theta, patient context vector
		theta = self.rn(input_tuple)

		# GRU layer
		seqs = torch.relu(self.fc1(seqs))
		seqs = self.dropout(seqs)
		packed_input = pack_padded_sequence(seqs, lengths, batch_first=True)
		rnn_output, _ = self.rnn(packed_input)
		padded_output, _ = pad_packed_sequence(rnn_output, batch_first=True)
		padded_output = padded_output[np.arange(len(padded_output)), lengths - 1]
		out = self.fc2(padded_output)

		# combine the patient context vector to the hidden state of the gru layer
		final_out = torch.add(out, theta)


		num_features = final_out.shape[-1]
		out = nn.Linear(num_features, self.num_classes)(final_out)
		return out

# recongition network to determine the patient context vector
class RecognitionNetwork(nn.Module):
	def __init__(self, input_dim, hidden_size, latent_dim):
		super(RecognitionNetwork, self).__init__()
		self.hidden_size = hidden_size
		self.latent_dim = latent_dim
		self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_size))
		self.fc2 = nn.Sequential(nn.Linear(hidden_size, hidden_size))
		self.fc3 = nn.Linear(hidden_size, 2 * latent_dim)  # output is 2*latent_dim to account for both mean and variance
		self.fc4 = nn.Linear(input_dim, latent_dim)

	def forward(self, input_tuple):
		x, seq_lengths = input_tuple

		# pack the padded sequence
		packed_sequence = pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)

		# apply the recognition network to the packed sequence
		x = torch.relu(self.fc1(packed_sequence.data))
		x = torch.relu(self.fc2(x))
		x = self.fc3(x)

		mu, logvar = torch.split(x, split_size_or_sections=self.latent_dim, dim=1)

		# pad the output mu and logvar tensors
		mu, _ = nn.utils.rnn.pad_packed_sequence(nn.utils.rnn.PackedSequence(mu, packed_sequence.batch_sizes),
												 batch_first=True)
		logvar, _ = nn.utils.rnn.pad_packed_sequence(nn.utils.rnn.PackedSequence(logvar, packed_sequence.batch_sizes),
													 batch_first=True)

		eps = torch.randn_like(logvar)
		theta = mu + logvar * eps

		l_b = self.fc4(packed_sequence.data)
		l_b, _ = nn.utils.rnn.pad_packed_sequence(nn.utils.rnn.PackedSequence(l_b, packed_sequence.batch_sizes),
												  batch_first=True)

		out = torch.mul(l_b, theta)

		out = torch.mean(out, dim=-1)
		return out

