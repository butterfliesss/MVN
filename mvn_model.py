""" MVN model for RTERC """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import Const


class AttentionModule(nn.Module):
	def __init__(self, hidden_size):
		super(AttentionModule, self).__init__()
		self.dropout = nn.Dropout(0.3)
		self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
		self.lin_1 = nn.Linear(hidden_size, hidden_size)
		self.lin_2 = nn.Linear(hidden_size, hidden_size)

		self.w_qs = nn.Linear(hidden_size, hidden_size, bias=False)
		self.w_ks = nn.Linear(hidden_size, hidden_size, bias=False)
		self.w_vs = nn.Linear(hidden_size, hidden_size, bias=False)

	def forward(self, q, k, v, attn_mask=None, use_residual=True, use_forward=True):
		residual = q
		q = self.w_qs(q)
		k = self.w_ks(k)
		v = self.w_vs(v)
		attn = torch.matmul(q, k.transpose(-2, -1))
		attn = attn * (k.size(-1) ** -0.5)

		if attn_mask is not None:
			attn.data.masked_fill_(attn_mask, -1e10)
		attn = F.softmax(attn, dim=-1)
		attn = self.dropout(attn)
		output = torch.matmul(attn, v)

		if use_residual:
			output += residual
			output = self.layer_norm(output)
			x1_res = output

		if use_forward:
			output = self.dropout(self.lin_2(F.relu(self.lin_1(output))))
			output += x1_res
			output = self.layer_norm(output)
		return output, attn


class GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(GatedFusion, self).__init__()
        self.lin_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_b = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, a, b):
        z = torch.sigmoid(self.lin_a(a) + self.lin_b(b))
        fusion_rep = z * a + (1 - z) * b
        return fusion_rep


class Max_Pool(nn.Module):
	def __init__(self):
		super(Max_Pool, self).__init__()
		self.dropout = nn.Dropout(0.3)

	def forward(self, inp_rep, mask):
		"""
		:param inp_rep: batch x len x dim
		:param mask: batch x len
		:return:
		"""
		inp_rep.data.masked_fill_(mask.unsqueeze(-1), -1e10)
		maxpl_first = inp_rep.max(dim=1)[0]
		maxpl_zero = torch.zeros_like(maxpl_first)
		maxpl = torch.where(maxpl_first != -1e10, maxpl_first, maxpl_zero)
		maxpl = self.dropout(maxpl)
		return maxpl


class GRUencoder(nn.Module):
	def __init__(self, d_emb, d_out, num_layers):
		super(GRUencoder, self).__init__()
		self.gru = nn.GRU(input_size=d_emb, hidden_size=d_out,
						  bidirectional=True, num_layers=num_layers)

	def forward(self, sent, sent_lens):
		"""
		:param sent: torch tensor, batch_size x len x dim
		:param sent_lens: numpy tensor, batch_size x 1
		:return:
		"""
		device = sent.device
		# seq_len x batch_size x d_rnn_in
		sent_embs = sent.transpose(0, 1)

		# sort by length
		s_lens, idx_sort = np.sort(sent_lens)[::-1].copy(), np.argsort(-sent_lens)
		# s_lens = s_lens.copy()
		idx_unsort = np.argsort(idx_sort)

		idx_sort = torch.from_numpy(idx_sort).cuda(device)
		s_embs = sent_embs.index_select(1, Variable(idx_sort))

		# padding
		sent_packed = pack_padded_sequence(s_embs, s_lens)
		sent_output = self.gru(sent_packed)[0]
		sent_output = pad_packed_sequence(sent_output, total_length=sent.size(1))[0]

		# unsort by length
		idx_unsort = torch.from_numpy(idx_unsort).cuda(device)
		sent_output = sent_output.index_select(1, Variable(idx_unsort))

		# batch x seq_len x 2*d_out
		output = sent_output.transpose(0, 1)
		return output


class MVN(nn.Module):
	def __init__(self, emodict, worddict, embedding, args):
		super(MVN, self).__init__()
		self.gpu = args.gpu
		self.device = torch.device("cuda:" + args.gpu)

		self.num_classes = emodict.n_words
		self.word2index = worddict.word2index
		self.index2word = worddict.index2word

		# sliding window
		self.hops = args.hops  
		self.wind_1 = args.wind1 

		self.embeddings = embedding
		self.utt_gru = GRUencoder(args.d_word_vec, args.d_h1, num_layers=1)
		
		self.d_lin_1 = args.d_h1 * 2
		self.cont_gru = nn.GRU(self.d_lin_1, self.d_lin_1, num_layers=1, bidirectional=True)
		
		self.dropout = nn.Dropout(0.3)
		
		self.max_pool = Max_Pool()

		# fusion gate
		self.fusion = GatedFusion(self.d_lin_1)
		self.word_level_fusion = GatedFusion(self.d_lin_1)
		self.utter_level_fusion = GatedFusion(self.d_lin_1)

		# attention module
		self.word_self_attn = AttentionModule(self.d_lin_1)
		self.word_cross_attn = AttentionModule(self.d_lin_1)

		# classifier
		# self.word_level_classifier = nn.Linear(self.d_lin_1, self.num_classes)
		# self.utter_level_classifier = nn.Linear(self.d_lin_1, self.num_classes)
 
		self.word_level_classifier = nn.Sequential(
		    nn.Linear(self.d_lin_1, args.d_h1),
		    nn.Tanh(),
		    nn.Dropout(0.3),
		    nn.Linear(args.d_h1, self.num_classes)
		)

		self.utter_level_classifier = nn.Sequential(
		    nn.Linear(self.d_lin_1, args.d_h1),
		    nn.Tanh(),
		    nn.Dropout(0.3),
		    nn.Linear(args.d_h1, self.num_classes)
		)
			
	def dotprod_attention(self, q, k, v, attn_mask=None):
		attn = torch.matmul(q, k.transpose(1, 2))
		if attn_mask is not None:
			attn.data.masked_fill_(attn_mask, -1e10)
		attn = F.softmax(attn, dim=-1)
		output = torch.matmul(attn, v)
		return output, attn

	def get_attn_pad_mask(self, seq_q, seq_k=None):
		if seq_k is None:
			assert seq_q.dim() == 2
			pad_attn_mask = seq_q.eq(Const.PAD)
		else:
			assert seq_q.dim() == 2 and seq_k.dim() == 2
			pad_attn_mask = torch.matmul(seq_q.unsqueeze(2).float(), seq_k.unsqueeze(1).float()).eq(Const.PAD)
		return pad_attn_mask.cuda(seq_q.device)

	def forward(self, sents, lengths):
		"""
		:param sents: batch x max_seq_len
		:param lengths: numpy array 1 x batch
		:return:
		"""
		device = sents.device
		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)
		
		# word embedding
		max_d_len = lengths.max().item() # real max len for each conv
		w_embed = self.dropout(self.embeddings(sents[:, :max_d_len])) 
		# batch(num of utters in a conv) x len x dim
		w_embed = self.utt_gru(w_embed, lengths)
		w_mask = torch.zeros((w_embed.size(0), w_embed.size(1))).cuda(device) # batch x len
		for i in range(w_embed.size(0)):
			w_mask[i, :lengths[i]] = 1

		w_attn_mask = self.get_attn_pad_mask(w_mask) # batch x len
		w_self_attn_mask = self.get_attn_pad_mask(w_mask, w_mask) # batch x len x len

		# word-level for first utter
		s_out_word_level = []
		first_query_self, _ = self.word_self_attn(w_embed[:1], w_embed[:1], w_embed[:1], w_self_attn_mask[:1]) # 1 x len x dim
		first_query_word_level_rep = self.max_pool(first_query_self, w_attn_mask[:1]) # first utter rep at word level, 1 x dim
		s_out_word_level.append(first_query_word_level_rep)

		# utterance-level for first utter
		s_out_utter_level = []
		s_utt = self.max_pool(w_embed, w_attn_mask) # utters rep, batch x dim
		first_query_utter_level_rep = s_utt[:1] # first utter rep at utter level, 1 x dim
		s_out_utter_level.append(first_query_utter_level_rep)

		cont_inp = s_utt.unsqueeze(1)  # batch x 1 x dim
		if sents.size()[0] > 1:
			# batch inputs and masks
			batches = []
			masks = []
			u_batches = []
			u_masks = []

			for i in range(1, sents.size()[0]):
				pad = max(self.wind_1 - i, 0)
				i_st = 0 if i < self.wind_1 + 1 else i - self.wind_1
				m_pad = F.pad(w_embed[i_st:i], (0, 0, 0, 0, pad, 0), mode='constant', value=0)  # context, K x len x dim
				m_pad = m_pad.unsqueeze(1)  # K x 1 x len x dim
				batches.append(m_pad)
				mask = torch.zeros(self.wind_1, w_embed.size(1)).long().to(sents.device)  # K x len
				if i_st == 0:
					for k in range(i_st, i):
						mask[pad + k, :lengths[k]] = 1
				else:
					for k in range(i_st, i):
						mask[k - i_st, :lengths[k]] = 1

				mask = mask.unsqueeze(1)  # K x 1 x len
				masks.append(mask)

				u_m_pad = F.pad(cont_inp[i_st:i], (0, 0, 0, 0, pad, 0), mode='constant', value=0)
				u_batches.append(u_m_pad)
				u_mask = [0] * pad + [1] * (self.wind_1 - pad)
				u_masks.append(u_mask)

			batches_tensor = torch.cat(batches, dim=1) # K x (batch-1) x len x dim
			masks_tensor = torch.cat(masks, dim=1)  # K x (batch-1) x len

			u_batches_tensor = torch.cat(u_batches, dim=1)  # K x (batch-1) x dim
			u_masks_tensor = torch.tensor(u_masks).long().to(sents.device)  # (batch-1) x K 

			u_query_mask = torch.ones(u_masks_tensor.size()[0], 1).long().to(sents.device)  # (batch-1) x 1
			u_attn_mask = self.get_attn_pad_mask(u_query_mask, u_masks_tensor)# (batch-1) x 1 x K
			
			# word-level view
			query_mask = w_mask[1:] # (batch-1) x len
			query_attn_mask = w_attn_mask[1:] # (batch-1) x len
			query_self_attn_mask = w_self_attn_mask[1:] # (batch-1) x len x len
			query = w_embed[1:].cuda(device)  # (batch-1) x len x dim

			context_bank = batches_tensor.transpose(0, 1).contiguous()  # (batch-1) x K x len x dim
			context_masks_tensor = masks_tensor.transpose(0, 1).contiguous()  # (batch-1) x K x len

			context_word = context_bank.reshape(context_bank.size(0), -1, context_bank.size(-1)) # (batch-1) x (Kxlen) x dim
			context_word_mask = context_masks_tensor.reshape(context_masks_tensor.size(0), -1)  # (batch-1) x (Kxlen)
			context_word_attn_mask = self.get_attn_pad_mask(context_word_mask)
			context_word_self_attn_mask = self.get_attn_pad_mask(context_word_mask, context_word_mask)
			query_context_word_cross_attn_mask = self.get_attn_pad_mask(query_mask, context_word_mask)
			# (batch-1) x len x (Kxlen)
			context_word_query_cross_attn_mask = self.get_attn_pad_mask(context_word_mask, query_mask)
			# (batch-1) x (Kxlen) x len
			
			# self-attention 
			query_self, _ = self.word_self_attn(query, query, query, query_self_attn_mask)
			context_word_self, _ = self.word_self_attn(context_word, context_word, context_word, 
			context_word_self_attn_mask)

			# cross-attention
			query_context_word_cross, _ = self.word_cross_attn(query, context_word, context_word, 
			query_context_word_cross_attn_mask)
			context_word_query_cross, _ = self.word_cross_attn(context_word, query, query, 
			context_word_query_cross_attn_mask)

			query_word_fusion = self.fusion(query_self, query_context_word_cross)
			query_word_rep = self.max_pool(query_word_fusion, query_attn_mask)

			context_word_fusion = self.fusion(context_word_self, context_word_query_cross)
			context_word_rep = self.max_pool(context_word_fusion, context_word_attn_mask)

			query_word_level_rep = self.word_level_fusion(query_word_rep, context_word_rep)
			s_out_word_level.append(query_word_level_rep)

			# utterance-level view
			query_utter_rep = s_utt[1:]
			mem_out = self.cont_gru(u_batches_tensor)[0]  # K x (batch-1) x dim
			mem_fwd, mem_bwd = mem_out.chunk(2, -1)
			context_utter_bank = (u_batches_tensor + mem_fwd + mem_bwd).transpose(0, 1).contiguous()  
			context_utter_bank = self.dropout(context_utter_bank)

			context_utter_rep, _ = self.dotprod_attention(query_utter_rep.unsqueeze(1),
			context_utter_bank, context_utter_bank, u_attn_mask)
			context_utter_rep = self.dropout(context_utter_rep.squeeze(1))

			query_utter_level_rep = self.utter_level_fusion(query_utter_rep, context_utter_rep)
			s_out_utter_level.append(query_utter_level_rep)

		s_cont_word_level = torch.cat(s_out_word_level, dim=0)
		s_output_word_level = self.word_level_classifier(s_cont_word_level)
		pred_s_word_level = F.log_softmax(s_output_word_level, dim=1)

		s_cont_utter_level = torch.cat(s_out_utter_level, dim=0)
		s_output_utter_level = self.utter_level_classifier(s_cont_utter_level)
		pred_s_utter_level = F.log_softmax(s_output_utter_level, dim=1)

		pred_s = pred_s_word_level + pred_s_utter_level

		return pred_s
