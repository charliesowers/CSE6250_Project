import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np

class RNN(nn.Module):
	def __init__(self, dim_input):
		super(RNN, self).__init__()
		# You may use the input argument 'dim_input', which is basically the number of features
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


### need work not working
class TopicRNN(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, num_topics):
		super(TopicRNN, self).__init__()
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.num_topics = num_topics

		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True, dropout=0.25)
		self.topic_dense = nn.Linear(hidden_dim	, num_topics)
		self.readmission_dense = nn.Linear(hidden_dim * num_topics, 2)

	def forward(self, x):
		embedded = self.embedding(x)
		output, _ = self.rnn(embedded)

		# Extract topic vector
		topic_vec = self.topic_dense(output[:, -1, :])

		# Softmax over topics to get topic distribution
		topic_dist = F.softmax(topic_vec, dim=1)

		# Weighted average of word embeddings to get topic representation
		weighted_emb = (embedded * topic_dist.unsqueeze(-1)).sum(dim=1)
		topic_rep = weighted_emb.view(-1, self.num_topics * self.embedding_dim)

		# Predict readmission
		readmission_scores = self.readmission_dense(topic_rep)
		return F.log_softmax(readmission_scores, dim=1)