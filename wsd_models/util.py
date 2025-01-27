#!/usr/bin/env python3
import os
import re
import torch
import subprocess
#from pytorch_transformers import *
# NEW in FEWS: pytorch_transformers --> transformers
from transformers import *
import random

pos_converter = {'NOUN':'n', 'PROPN':'n', 'VERB':'v', 'AUX':'v', 'ADJ':'a', 'ADV':'r'}

def score(preds, labels):
	# NEW in FEWS: calculate accuracy
	print(len(preds))
	print(len(labels))
	total_preds = len(preds)
	num_correct = 0.
	for p, l in zip(preds, labels):
		if p == l:
			num_correct += 1
	acc = num_correct/total_preds
	return acc

def generate_key(lemma, pos):
	if pos in pos_converter.keys():
		pos = pos_converter[pos]
	key = '{}+{}'.format(lemma, pos)
	return key

def load_pretrained_model(name):
	# NEW in FEWS: add XLMR
	if name == 'xlmr-base':
		model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
		hdim = 768
	elif name == 'xlmr-large':
		model = XLMRobertaModel.from_pretrained('xlm-roberta-large')
		hdim = 1024
	elif name == 'roberta-base':
		model = RobertaModel.from_pretrained('roberta-base')
		hdim = 768
	elif name == 'roberta-large':
		model = RobertaModel.from_pretrained('roberta-large')
		hdim = 1024
	elif name == 'bert-large':
		model = BertModel.from_pretrained('bert-large-uncased')
		hdim = 1024
	else: #bert base
		model = BertModel.from_pretrained('bert-base-uncased')
		hdim = 768
	return model, hdim

def load_tokenizer(name):
	# NEW in FEWS:added XLMR, swithed from BERT uncased to cased
	if name == 'xlmr-base':
		tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
	elif name == 'xlmr-large':
		tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
	elif name == 'roberta-base':
		tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
	elif name == 'roberta-large':
		tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
	elif name == 'bert-large':
		tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
	else: #bert base
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	return tokenizer

def load_wn_senses(path):
	wn_senses = {}
	with open(path, 'r', encoding="utf8") as f:
		for line in f:
			line = line.strip().split('\t')
			lemma = line[0]
			pos = line[1]
			senses = line[2:]

			key = generate_key(lemma, pos)
			wn_senses[key] = senses
	return wn_senses

def load_fews_senses(path):
	# NEW in FEWS: returns a map from (lemma,pos) to (list of senseids, list of glosses)
	senses = {}
	with open(path, 'r') as f:
		s = {}
		for line in f:
			line = line.strip()
			if len(line) == 0:
				key = '.'.join(s['sense_id'].split('.')[:2])
				if key in senses:
					key_arr, gloss_arr = senses[key]
					key_arr.append(s['sense_id'])
					gloss_arr.append(s['gloss'])
					senses[key] = (key_arr, gloss_arr)
				else:
					senses[key] = ([s['sense_id']],[s['gloss']])
				s = {}
			else:
				line = line.strip().split(':\t')
				key = line[0]
				if len(line) > 1: value = line[1]
				else:
					key = key[:-1]
					value = ''
				s[key] = value
		#deal with last sense
		if len(s.keys()) > 0:
			key = '.'.join(s['sense_id'].split('.')[:2])
			if key in senses:
				key_arr, gloss_arr = senses[key]
				key_arr.append(s['sense_id'])
				gloss_arr.append(s['gloss'])
				senses[key] = (key_arr, gloss_arr)
			else:
				senses[key] = ([s['sense_id']],[s['gloss']])
	return senses

def get_label_space(data, dataset='wn'):
	# NEW in FEWS: not sure, but looks data is iterable of (usage,label) and we return a map from (lemma,pos) to senses we meet among usages
	if dataset=='fews':
		labels, label_map = _fews_label_space(data)
	else:
		labels, label_map = _unified_label_space(data)
	return labels, label_map

def _fews_label_space(data):
	# NEW in FEWS:
	#get set of labels from dataset
	labels = set()
	
	for d in data:
		_, label = d
		labels.add(label)

	labels = list(labels)
	labels.sort()
	labels.append('n/a')

	label_map = {}
	for d in data:
		sent, label = d
		key = '.'.join(label.split('.')[:2])
		label_idx = labels.index(label)
		if key not in label_map: label_map[key] = set()
		label_map[key].add(label_idx)

	return labels, label_map

def _unified_label_space(data):
	# NEW in FEWS: function get_label_space renamed to _unified_label_space, code is identical
	#get set of labels from dataset
	labels = set()
	
	for sent in data:
		for _, _, _, _, label in sent:
			if label != -1:
				labels.add(label)

	labels = list(labels)
	labels.sort()
	labels.append('n/a')

	label_map = {}
	for sent in data:
		for _, lemma, pos, _, label in sent:
			if label != -1:
				key = generate_key(lemma, pos)
				label_idx = labels.index(label)
				if key not in label_map: label_map[key] = set()
				label_map[key].add(label_idx)

	return labels, label_map

def process_encoder_outputs(output, mask, as_tensor=False):
	combined_outputs = []
	position = -1
	avg_arr = []
	for idx, rep in zip(mask, torch.split(output, 1, dim=0)):
		#ignore unlabeled words
		if idx == -1: continue
		#average representations for units in same example
		elif position < idx: 
			position=idx
			if len(avg_arr) > 0: combined_outputs.append(torch.mean(torch.stack(avg_arr, dim=-1), dim=-1))
			avg_arr = [rep]
		else:
			assert position == idx 
			avg_arr.append(rep)
	#get last example from avg_arr
	if len(avg_arr) > 0: combined_outputs.append(torch.mean(torch.stack(avg_arr, dim=-1), dim=-1))
	if as_tensor: return torch.cat(combined_outputs, dim=0)
	else: return combined_outputs

#run WSD Evaluation Framework scorer within python
def evaluate_output(scorer_path, gold_filepath, out_filepath):
	eval_cmd = ['java','-cp', scorer_path, 'Scorer', gold_filepath, out_filepath]
	output = subprocess.Popen(eval_cmd, stdout=subprocess.PIPE ).communicate()[0]
	output = [x.decode("utf-8") for x in output.splitlines()]
	p,r,f1 =  [float(output[i].split('=')[-1].strip()[:-1]) for i in range(3)]
	return p, r, f1

def load_data(datapath, name, dataset='wn'):
	# NEW in FEWS:
	if dataset == 'fews':
		data = _load_fews_data(datapath, name)
	else: 
		data = _load_unified_data(datapath, name)
	return data

def _load_fews_data(datapath, name):
	# NEW in FEWS: load FEWS usages
	text_path = os.path.join(datapath, '{}.txt'.format(name))
	'''
	This function is adapted from the data_utils.py script
	provided with the Few-Shot WSD dataset
	'''
	examples = []
	with open(text_path, 'r') as f:
		for line in f:
			sent, label = line.strip().split('\t')
			examples.append((sent, label))
	return examples

def _load_unified_data(datapath, name):
	# NEW in FEWS: name changed from load_data, this loads Raganato's unified datasets
	text_path = os.path.join(datapath, '{}.data.xml'.format(name))
	gold_path = os.path.join(datapath, '{}.gold.key.txt'.format(name))

	#load gold labels 
	gold_labels = {}
	with open(gold_path, 'r', encoding="utf8") as f:
		for line in f:
			line = line.strip().split(' ')
			instance = line[0]
			#this means we are ignoring other senses if labeled with more than one 
			#(happens at least in SemCor data)
			key = line[1]
			gold_labels[instance] = key

	#load train examples + annotate sense instances with gold labels
	sentences = []
	s = []
	with open(text_path, 'r', encoding="utf8") as f:
		for line in f:
			line = line.strip()
			if line == '</sentence>':
				sentences.append(s)
				s=[]
				
			elif line.startswith('<instance') or line.startswith('<wf'):
				word = re.search('>(.+?)<', line).group(1)
				lemma = re.search('lemma="(.+?)"', line).group(1) 
				pos =  re.search('pos="(.+?)"', line).group(1)

				#clean up data
				word = re.sub('&apos;', '\'', word)
				lemma = re.sub('&apos;', '\'', lemma)

				sense_inst = -1
				sense_label = -1
				if line.startswith('<instance'):
					sense_inst = re.search('instance id="(.+?)"', line).group(1)
					#annotate sense instance with gold label
					sense_label = gold_labels[sense_inst]
				s.append((word, lemma, pos, sense_inst, sense_label))

	return sentences

#normalize ids list, masks to whatever the passed in length is
def normalize_length(ids, attn_mask, o_mask, max_len, pad_id):
	if max_len == -1:
		return ids, attn_mask, o_mask
	else:
		if len(ids) < max_len:
			while len(ids) < max_len:
				ids.append(torch.tensor([[pad_id]]))
				attn_mask.append(0)
				o_mask.append(-1)
		else:
			ids = ids[:max_len-1]+[ids[-1]]
			attn_mask = attn_mask[:max_len]
			o_mask = o_mask[:max_len]

		assert len(ids) == max_len
		assert len(attn_mask) == max_len
		assert len(o_mask) == max_len

		return ids, attn_mask, o_mask

#filters down training dataset to (up to) k examples per sense 
#for few-shot learning of the model
def filter_k_examples(data, k):
	#shuffle data so we don't only get examples for (common) senses from beginning
	random.shuffle(data)
	#track number of times sense from data is used
	sense_dict = {}
	#store filtered data 
	filtered_data = []

	example_count = 0
	for sent in data:
		filtered_sent = []
		for form, lemma, pos, inst, sense in sent:
			#treat unlabeled words normally
			if sense == -1:
				x  = (form, lemma, pos, inst, sense)
			elif sense in sense_dict:
				if sense_dict[sense] < k: 
					#increment sense count and add example to filtered data
					sense_dict[sense] += 1
					x = (form, lemma, pos, inst, sense)
					example_count += 1
				else: #if the data already has k examples of this sense
					#add example with no instance or sense label to data
					x = (form, lemma, pos, -1, -1)
			else:
				#add labeled example to filtered data and sense dict
				sense_dict[sense] = 1
				x = (form, lemma, pos, inst, sense)
				example_count += 1
			filtered_sent.append(x)
		filtered_data.append(filtered_sent)

	print("k={}, training on {} sense examples...".format(k, example_count))

	return filtered_data

#EOF