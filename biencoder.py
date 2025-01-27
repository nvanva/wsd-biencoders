#!/usr/bin/env python3
import torch
from torch.nn import functional as F
from nltk.corpus import wordnet as wn
import os
import sys
import time
import math
import copy
import argparse
from tqdm import tqdm
import pickle
from collections import OrderedDict
#from pytorch_transformers import *
# NEW in FEWS: switched from pytorch_transformers to transformers, tensorboardX is now imported!
from transformers import *
from tensorboardX import SummaryWriter

import random
import numpy as np

from wsd_models.util import *
from wsd_models.models import BiEncoderModel

import traceback

# NEW in FEWS: hard-coded paths to FEWS and Raganato's datasets - get rid of!
WN_DATAPATH = '/checkpoint/tblevins/data/wsd_framework/'
FEWS_DATAPATH = './fews/'

#SAMPLE_N_FROM_TRAIN = 1500 #total num sentences in test is 1173

parser = argparse.ArgumentParser(description='Gloss Informed Bi-encoder for WSD')
parser.add_argument('--device', type=str, default='cuda:0')

#training arguments
parser.add_argument('--rand_seed', type=int, default=42)
parser.add_argument('--grad-norm', type=float, default=1.0)
parser.add_argument('--silent', action='store_true',
	help='Flag to supress training progress bar for each epoch')
parser.add_argument('--multigpu', action='store_true')
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--context-max-length', type=int, default=128)
parser.add_argument('--gloss-max-length', type=int, default=32)
parser.add_argument('--epochs', type=int, default=20)
# NEW in FEWS: context batch size increased from 4 to 8
parser.add_argument('--context-bsz', type=int, default=8)
parser.add_argument('--gloss-bsz', type=int, default=256)
# NEW in FEWS: Terra added XLMR as one of the backbones (in parallel to us?)
parser.add_argument('--encoder-name', type=str, default='bert-base',
	choices=['bert-base', 'bert-large', 'roberta-base', 'roberta-large', 'xlmr-base', 'xlmr-large'])
parser.add_argument('--ckpt', type=str, required=True,
	help='filepath at which to save best probing model (on dev set)')
parser.add_argument('--nonstrict_load', action='store_true',
	help='Do not enforce that all keys match between the loaded checkpoint and the model, required to load older checkpoints without embeddings.position_ids (which are not weights but just a non-learnt tensor with range(0,512), so no need to load them). But this can lead to some weights staying random / non-finetuned. Start with strict load and check which tensors are not loaded, if required set this to true if you absolutely sure they should not be loaded.')

# NEW in FEWS: --data-path is replaced with --dataset
#parser.add_argument('--data-path', type=str, required=True,
#	help='Location of top-level directory for the WordNet Unified WSD Framework or FEWS dataset')
parser.add_argument('--dataset', type=str, default='wn',
	choices=['wn', 'fews'],
	help='Which dataset type to train/eval model on')
# NEW in FEWS: training on the extended FEWS train set (includes example usages authored by Witionary authors)
parser.add_argument('--fews-extended-train', action='store_true',
	help='Train on extended training data on fews dataset')
# NEW in FEWS: fine-tuning an already trained model (for fine-tuning of FEWS a model trained on SemCor and vice versa?)
parser.add_argument('--load-existing-ckpt', type=str, default='',
	help='Initializes model with saved biencoder weights from given path instead of intializing with vanilla BERT')

#sets which parts of the model to freeze ❄️ during training for ablation 
parser.add_argument('--freeze-context', action='store_true')
parser.add_argument('--freeze-gloss', action='store_true')
parser.add_argument('--tie-encoders', action='store_true')
# NEW in FEWS: removed --kshot (limiting to k training examples per sense), --balanced (reweighting sense losses) arguments

#evaluation arguments
parser.add_argument('--eval', action='store_true',
	help='Flag to set script to evaluate probe (rather than train)')
# NEW in FEWS: the default dataset in Raganato's (dev set?) changed from se2007 to se2015 for some reason...
parser.add_argument('--wn-split', type=str, default='semeval2015',
	choices=['semeval2007', 'senseval2', 'senseval3', 'semeval2013', 'semeval2015', 'ALL', 'all-test'],
	help='Which Unified Framework evaluation split on which to evaluate probe')
# NEW in FEWS:
parser.add_argument('--fews-split', type=str, default='dev',
	choices=['dev', 'train', 'train_ext', 'test', 'dev-human-subset'],
	help='Which Few-Shot WSD evaluation split on which to evaluate probe')

#uses these two gpus if training in multi-gpu
context_device = "cuda:0"
gloss_device = "cuda:1"

def tokenize_glosses(gloss_arr, tokenizer, max_len):
	glosses = []
	masks = []
	for gloss_text in gloss_arr:
		# NEW in FEWS: disabled adding special tokens (seems like Maxim did the same thing when moving to XLMR, why?)
		g_ids = [torch.tensor([[x]]) for x in tokenizer.encode(tokenizer.cls_token, add_special_tokens=False)+tokenizer.encode(gloss_text, add_special_tokens=False)+tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)]
		g_attn_mask = [1]*len(g_ids)
		g_fake_mask = [-1]*len(g_ids)
		g_ids, g_attn_mask, _ = normalize_length(g_ids, g_attn_mask, g_fake_mask, max_len, pad_id=tokenizer.encode(tokenizer.pad_token)[0])
		g_ids = torch.cat(g_ids, dim=-1)
		g_attn_mask = torch.tensor(g_attn_mask)
		glosses.append(g_ids)
		masks.append(g_attn_mask)

	return glosses, masks

#creates a sense label/ gloss dictionary for training/using the gloss encoder
def preprocess_wn_glosses(data, tokenizer, wn_senses, max_len=-1):
	# NEW in FEWS: renamed from load_and_preprocess_glosses, code is almost identical but  sense_weights is not built and returned anymore
	sense_glosses = {}
	gloss_lengths = []

	for sent in data:
		for _, lemma, pos, _, label in sent:
			if label == -1:
				continue #ignore unlabeled words
			else:
				key = generate_key(lemma, pos)
				if key not in sense_glosses:
					#get all sensekeys for the lemma/pos pair
					sensekey_arr = wn_senses[key]
					#get glosses for all candidate senses
					gloss_arr = [wn.lemma_from_key(s).synset().definition() for s in sensekey_arr]

					#preprocess glosses into tensors
					gloss_ids, gloss_masks = tokenize_glosses(gloss_arr, tokenizer, max_len)
					gloss_ids = torch.cat(gloss_ids, dim=0)
					gloss_masks = torch.stack(gloss_masks, dim=0)
					sense_glosses[key] = (gloss_ids, gloss_masks, sensekey_arr)
				
				#make sure that gold label is retrieved synset
				assert label in sense_glosses[key][2]

	return sense_glosses

def preprocess_fews_glosses(data, tokenizer, senses, max_len=-1):
	# NEW in FEWS: new function
	sense_glosses = {}
	gloss_lengths = []

	for sent, label in data:	
		key = '.'.join(label.split('.')[:2])
		if key not in sense_glosses:
			#get all sensekeys for the lemma/pos pair
			key_arr, gloss_arr = senses[key]

			#clean gloss text (remove brackets)
			gloss_arr = [g.replace('[[', '').replace(']]', '') for g in gloss_arr]
			#preprocess glosses into tensors
			gloss_ids, gloss_masks = tokenize_glosses(gloss_arr, tokenizer, max_len)
			gloss_ids = torch.cat(gloss_ids, dim=0)
			gloss_masks = torch.stack(gloss_masks, dim=0)
			sense_glosses[key] = (gloss_ids, gloss_masks, key_arr)
		
		#make sure that gold label is retrieved synset
		assert label in sense_glosses[key][2]

	return sense_glosses

def preprocess_wn_context(tokenizer, text_data, bsz=1, max_len=-1):
	# NEW in FEWS: renamed from process_context, code is identical but added add_special_tokens=False to all tokenizer.encode() calls
	if max_len == -1: assert bsz==1 #otherwise need max_length for padding

	context_ids = []
	context_attn_masks = []

	example_keys = []

	context_output_masks = []
	instances = []
	labels = []

	#tensorize data
	for sent in text_data:
		c_ids = [torch.tensor([tokenizer.encode(tokenizer.cls_token, add_special_tokens=False)])] #cls token aka sos token, returns a list with index
		o_masks = [-1]
		sent_insts = []
		sent_keys = []
		sent_labels = []

		#For each word in sentence...
		for idx, (word, lemma, pos, inst, label) in enumerate(sent):
			#tensorize word for context ids
			#print(word)
			word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word, add_special_tokens=False)]
			c_ids.extend(word_ids)

			#print(c_ids)
			#input('...')
			#if word is labeled with WSD sense...
			if inst != -1:
				#add word to bert output mask to be labeled
				o_masks.extend([idx]*len(word_ids))
				#track example instance id
				sent_insts.append(inst)
				#track example instance keys to get glosses
				ex_key = generate_key(lemma, pos)
				sent_keys.append(ex_key)
				sent_labels.append(label)
			else:
				#mask out output of context encoder for WSD task (not labeled)
				o_masks.extend([-1]*len(word_ids))

			#break if we reach max len
			if max_len != -1 and len(c_ids) >= (max_len-1):
				break

		c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)])) #aka eos token
		c_attn_mask = [1]*len(c_ids)
		o_masks.append(-1)
		c_ids, c_attn_masks, o_masks = normalize_length(c_ids, c_attn_mask, o_masks, max_len, pad_id=tokenizer.encode(tokenizer.pad_token)[0])

		y = torch.tensor([1]*len(sent_insts), dtype=torch.float)
		#not including examples sentences with no annotated sense data
		if len(sent_insts) > 0:
			context_ids.append(torch.cat(c_ids, dim=-1))
			context_attn_masks.append(torch.tensor(c_attn_masks).unsqueeze(dim=0))
			context_output_masks.append(torch.tensor(o_masks).unsqueeze(dim=0))
			example_keys.append(sent_keys)
			instances.append(sent_insts)
			labels.append(sent_labels)

	#package data
	data = list(zip(context_ids, context_attn_masks, context_output_masks, example_keys, instances, labels))

	#batch data if bsz > 1
	if bsz > 1:
		print('Batching data with bsz={}...'.format(bsz))
		batched_data = []
		for idx in range(0, len(data), bsz):
			if idx+bsz <=len(data): b = data[idx:idx+bsz]
			else: b = data[idx:]
			context_ids = torch.cat([x for x,_,_,_,_,_ in b], dim=0)
			context_attn_mask = torch.cat([x for _,x,_,_,_,_ in b], dim=0)
			context_output_mask = torch.cat([x for _,_,x,_,_,_ in b], dim=0)
			example_keys = []
			for _,_,_,x,_,_ in b: example_keys.extend(x)
			instances = []
			for _,_,_,_,x,_ in b: instances.extend(x)
			labels = []
			for _,_,_,_,_,x in b: labels.extend(x)
			batched_data.append((context_ids, context_attn_mask, context_output_mask, example_keys, instances, labels))
		return batched_data
	else:  
		return data

def preprocess_fews_context(tokenizer, text_data, bsz=1, max_len=-1):
	# NEW in FEWS:
	if max_len == -1: 
		assert bsz==1 #otherwise need max_length for padding
		max_len = 512
	context_ids = []
	context_attn_masks = []

	example_keys = []

	context_output_masks = []
	instances = []
	labels = []

	skip_count = 0

	#tensorize data
	inst = 0
	for sent, label in text_data:
		sent_inst_id = 1
		#input ids for sentence
		c_ids = [torch.tensor([tokenizer.encode(tokenizer.cls_token, add_special_tokens=False)])] #cls token aka sos token, returns a list with index
		#bert output mask
		o_masks = [-1]
		sent_insts = []
		sent_keys = []
		sent_labels = []

		sent_pieces = sent.split('<WSD>')
		for s in sent_pieces:
			if '</WSD>' in s:
				s_inst, s = s.split('</WSD>')

				#process labeled example
				word_ids = [torch.tensor([t]).unsqueeze(0) for t in tokenizer.encode(s_inst, add_special_tokens=False)]

				c_ids.extend(word_ids)
				o_masks.extend([sent_inst_id]*len(word_ids))
				sent_inst_id += 1
				sent_insts.append(inst)
				inst += 1
				
				#track example instance keys to get glosses
				ex_key = '.'.join(label.split('.')[:2])
				sent_keys.append(ex_key)
				sent_labels.append(label)

				#process other text
				word_ids = [torch.tensor([t]).unsqueeze(0) for t in tokenizer.encode(s)]
				c_ids.extend(word_ids)
				o_masks.extend([-1]*len(word_ids))
			else:
				#process sentence text
				word_ids = [torch.tensor([t]).unsqueeze(0) for t in tokenizer.encode(s, add_special_tokens=False)]
				c_ids.extend(word_ids)
				o_masks.extend([-1]*len(word_ids))

		c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)])) #aka eos token
		c_attn_mask = [1]*len(c_ids)
		o_masks.append(-1)
		c_ids, c_attn_masks, o_masks = normalize_length(c_ids, c_attn_mask, o_masks, max_len, pad_id=tokenizer.encode(tokenizer.pad_token)[0])

		#remove examples in truncated part of sentences
		num_examples = len(set([m for m in o_masks if m != -1]))
		sent_keys = sent_keys[:num_examples]
		sent_labels = sent_labels[:num_examples]

		y = torch.tensor([1]*len(sent_insts), dtype=torch.float)
		#not including examples sentences with no annotated sense data

		if sum(o_masks) > (max_len*-1):
			context_ids.append(torch.cat(c_ids, dim=-1))
			context_attn_masks.append(torch.tensor(c_attn_masks).unsqueeze(dim=0))
			context_output_masks.append(torch.tensor(o_masks).unsqueeze(dim=0))
			example_keys.append(sent_keys)
			instances.append(sent_insts)
			labels.append(sent_labels)
		else:
			skip_count += 1

	print('Skipped {} sentences (no examples in length)'.format(skip_count))

	#package data
	data = list(zip(context_ids, context_attn_masks, context_output_masks, example_keys, instances, labels))

	#batch data if bsz > 1
	if bsz > 1:
		print('Batching data with bsz={}...'.format(bsz))
		batched_data = []
		for idx in range(0, len(data), bsz):
			if idx+bsz <=len(data): b = data[idx:idx+bsz]
			else: b = data[idx:]
			context_ids = torch.cat([x for x,_,_,_,_,_ in b], dim=0)
			context_attn_mask = torch.cat([x for _,x,_,_,_,_ in b], dim=0)
			context_output_mask = torch.cat([x for _,_,x,_,_,_ in b], dim=0)
			example_keys = []
			for _,_,_,x,_,_ in b: example_keys.extend(x)
			instances = []
			for _,_,_,_,x,_ in b: instances.extend(x)
			labels = []
			for _,_,_,_,_,x in b: labels.extend(x)
			batched_data.append((context_ids, context_attn_mask, context_output_mask, example_keys, instances, labels))
		return batched_data
	else:  
		return data

def _train(train_data, model, gloss_dict, optim, schedule, criterion, gloss_bsz=-1, max_grad_norm=1.0, multigpu=False, silent=False, train_steps=-1):
	model.train()
	total_loss = 0.

	start_time = time.time()

	train_data = enumerate(train_data)
	if not silent: train_data = tqdm(list(train_data))

	for i, (context_ids, context_attn_mask, context_output_mask, example_keys, _, labels) in train_data:

		#reset model
		model.zero_grad()
		#run example sentence(s) through context encoder
		# if multigpu:
		context_ids = context_ids.to(context_device)
		context_attn_mask = context_attn_mask.to(context_device)
		# else:
		# 	context_ids = context_ids.cuda()
		# 	context_attn_mask = context_attn_mask.cuda()
		context_output = model.context_forward(context_ids, context_attn_mask, context_output_mask)

		loss = 0.
		gloss_sz = 0
		context_sz = len(labels)
		for j, (key, label) in enumerate(zip(example_keys, labels)):
			output = context_output.split(1,dim=0)[j]

			#run example's glosses through gloss encoder
			gloss_ids, gloss_attn_mask, sense_keys = gloss_dict[key]
			# if multigpu:
			gloss_ids = gloss_ids.to(gloss_device)
			gloss_attn_mask = gloss_attn_mask.to(gloss_device)
			# else:
			# 	gloss_ids = gloss_ids.cuda()
			# 	gloss_attn_mask = gloss_attn_mask.cuda()

			gloss_output = model.gloss_forward(gloss_ids, gloss_attn_mask)
			gloss_output = gloss_output.transpose(0,1)
			
			#get cosine sim of example from context encoder with gloss embeddings
			if multigpu:
				output = output.cpu()
				gloss_output = gloss_output.cpu()
			
			output = torch.mm(output, gloss_output)

			#get label and calculate loss
			idx = sense_keys.index(label)
			label_tensor = torch.tensor([idx])
			if not multigpu: label_tensor = label_tensor.cuda()

			#looks up correct candidate senses criterion
			#needed if balancing classes within the candidate senses of a target word
			# NEW in FEWS: criterion[key] is replaced to a single criterion, looks like no weighting depending on sense frequencies now
			loss += criterion(output, label_tensor)
			gloss_sz += gloss_output.size(-1)

			if gloss_bsz != -1 and gloss_sz >= gloss_bsz:
				#update model
				total_loss += loss.item()
				loss=loss/gloss_sz
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
				optim.step()
				schedule.step() # Update learning rate schedule

				#reset loss and gloss_sz
				loss = 0.
				gloss_sz = 0

				#reset model
				model.zero_grad()

				#rerun context through model
				context_output = model.context_forward(context_ids, context_attn_mask, context_output_mask)

		#update model after finishing context batch
		if gloss_bsz != -1: loss_sz = gloss_sz
		else: loss_sz = context_sz
		if loss_sz > 0:
			total_loss += loss.item()
			loss=loss/loss_sz
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
			optim.step()
			schedule.step() # Update learning rate schedule

		#stop epoch early if number of training steps is reached
		if train_steps > 0 and i+1 == train_steps: break

	return model, optim, schedule, total_loss

def _eval(eval_data, model, gloss_dict, multigpu=False):
	model.eval()
	eval_preds = []
	for context_ids, context_attn_mask, context_output_mask, example_keys, insts, _ in tqdm(eval_data):
		with torch.no_grad(): 
			#run example through model
			# if multigpu:
			context_ids = context_ids.to(context_device)
			context_attn_mask = context_attn_mask.to(context_device)
			# else:
			# 	context_ids = context_ids.cuda()
			# 	context_attn_mask = context_attn_mask.cuda()
			context_output = model.context_forward(context_ids, context_attn_mask, context_output_mask)

			for output, key, inst in zip(context_output.split(1,dim=0), example_keys, insts):
				#run example's glosses through gloss encoder
				gloss_ids, gloss_attn_mask, sense_keys = gloss_dict[key]
				# if multigpu:
				gloss_ids = gloss_ids.to(gloss_device)
				gloss_attn_mask = gloss_attn_mask.to(gloss_device)
				# else:
				# 	gloss_ids = gloss_ids.cuda()
				# 	gloss_attn_mask = gloss_attn_mask.cuda()
				gloss_output = model.gloss_forward(gloss_ids, gloss_attn_mask)
				gloss_output = gloss_output.transpose(0,1)

				#get cosine sim of example from context encoder with gloss embeddings
				if multigpu:
					output = output.cpu()
					gloss_output = gloss_output.cpu()
				output = torch.mm(output, gloss_output)
				pred_idx = output.topk(1, dim=-1)[1].squeeze().item()
				pred_label = sense_keys[pred_idx]
				eval_preds.append((inst, pred_label))

	return eval_preds

def data_loading(args, tokenizer, eval='', process_train=True):
	# NEW in FEWS: new data loading function, extracted from train_model() and generalized to fews
	dataset = args.dataset
	#set datapath to use
	if dataset == 'fews': datapath = FEWS_DATAPATH
	else: datapath = WN_DATAPATH

	#set default eval split if none given
	if eval == '':
		if dataset == 'fews': eval = 'dev'
		else: eval = 'semeval2015'

	#load senses
	if dataset == 'fews':
		s_path = os.path.join(datapath, 'senses.txt')
		senses = load_fews_senses(s_path)
	else:
		wn_path = os.path.join(datapath, 'Data_Validation/candidatesWN30.txt')
		senses = load_wn_senses(wn_path)

	train_data = -1
	train_glosses = -1
	if process_train:
		#read in train data
		if dataset == 'fews':
			#read in extended train data if flag is passed in
			train_path = os.path.join(datapath, 'train')
			if args.fews_extended_train == True:
				train_data = load_data(train_path, 'train.ext', dataset=dataset)
			else:
				train_data = load_data(train_path, 'train', dataset=dataset)
		else:
			train_path = os.path.join(datapath, 'Training_Corpora/SemCor')
			train_data = load_data(train_path, 'semcor', dataset=dataset)

		#DEBUGGING
		#train_data = train_data[:1000]

		# get train glosses + contexts
		if dataset == 'fews':
			train_glosses = preprocess_fews_glosses(train_data, tokenizer, senses, max_len=args.gloss_max_length)
			train_data = preprocess_fews_context(tokenizer, train_data, bsz=args.context_bsz, max_len=args.context_max_length)
		else:
			train_glosses = preprocess_wn_glosses(train_data, tokenizer, senses, max_len=args.gloss_max_length)
			train_data = preprocess_wn_context(tokenizer, train_data, bsz=args.context_bsz, max_len=args.context_max_length)

	#load eval_data
	if dataset == 'fews':
		if 'dev' in eval:
			eval_path = os.path.join(datapath, 'dev')
		else:
			eval_path = os.path.join(datapath, 'test')
		eval_data = load_data(eval_path, eval+'.few-shot', dataset=dataset)+load_data(eval_path, eval+'.zero-shot', dataset=dataset)
	elif eval != 'train_subset':
		eval_path = os.path.join(datapath, 'Evaluation_Datasets/{}/'.format(eval))
		eval_data = load_data(eval_path, eval, dataset=dataset)

	#get eval glosses
	if dataset == 'fews':
		eval_glosses = preprocess_fews_glosses(eval_data, tokenizer, senses, max_len=args.gloss_max_length)
		eval_data = preprocess_fews_context(tokenizer, eval_data, bsz=1, max_len=-1)
	else:
		eval_glosses = preprocess_wn_glosses(eval_data, tokenizer, senses, max_len=args.gloss_max_length)
		eval_data = preprocess_wn_context(tokenizer, eval_data, bsz=1, max_len=-1)
		
	return train_data, eval_data, train_glosses, eval_glosses

def generate_gold_file(ckpt, data):
	# NEW in FEWS:
	gold_filepath = os.path.join(ckpt, 'train_subset_gold.txt')
	with open(gold_filepath, 'w') as f:
		for _, _, _, _, insts, labels in data:
			for inst, label in zip(insts, labels):
				f.write('{} {}\n'.format(inst, label))
	return


def model_loading(args, eval_mode=False):
	model = BiEncoderModel(args.encoder_name, freeze_gloss=args.freeze_gloss, freeze_context=args.freeze_context, tie_encoders=args.tie_encoders)

	if eval_mode or args.load_existing_ckpt:
		model_path = os.path.join(args.ckpt if eval_mode else args.load_existing_ckpt, 'best_model.ckpt')

		try:
			state_dict = torch.load(model_path, map_location=context_device, weights_only=True)
			# old checkpoints don't have position_ids tensor dumped;
			# since this is just non-learnt range(0,512), it is safe to take them from (even a randomly initialized) model
			for enc in ('context_encoder', 'gloss_encoder'):
				k = f'{enc}.{enc}.embeddings.position_ids'
				if k in model.state_dict() and not k in state_dict:
					state_dict[k] = model.state_dict()[k]
			model.load_state_dict(state_dict, strict=not args.nonstrict_load)
			print('Model weights successfully loaded from', model_path)
		except RuntimeError: #this happens with other models that we justz want encoder from
			print(f"Problems when loading the checkpoint from {model_path}:\n", traceback.format_exc(), file=sys.stderr)
			print(f"Trying to recover by renaming weights...", file=sys.stderr)
			#Get and filter state dict
			state_dict = torch.load(model_path, weights_only=True)
			new_state_dict = OrderedDict()
			for k, v in state_dict.items():
				name = k
				if name.startswith('module.'): #cleaning up from previous multigpu (if still wrapped)
					name = name[7:] # remove `module.`
				if name.startswith('encoder.'): #and (name.endswith('.weight') or name.endswith('.bias')):
					name = name[8:] #remove `encoder.`
					new_state_dict[name] = v
			state_dict = new_state_dict
			#not_overlap = []
			#for k in state_dict.keys():
			#	if k not in model.context_encoder.context_encoder.state_dict().keys():
			#		not_overlap.append(k)
			#print(not_overlap)
			#quit()
			#load context_encoder states
			model.context_encoder.context_encoder.load_state_dict(state_dict, strict=not args.nonstrict_load)
			#load gloss_encoder states
			model.gloss_encoder.gloss_encoder.load_state_dict(state_dict, strict=not args.nonstrict_load)

	#speeding up training by putting two encoders on seperate gpus (instead of data parallel)
	# if args.multigpu:
	model.gloss_encoder = model.gloss_encoder.to(gloss_device)
	model.context_encoder = model.context_encoder.to(context_device)
	# else:
	# 	model = model.cuda()
	return model


def train_model(args):
	# NEW in FEWS: many changes here: using tensorborad, warm start from a specified checkpoint, evaluating on se2015
	# when training on Raganato's, on FEWS using accuracy instead of f1,
	if args.freeze_gloss: assert args.gloss_bsz == -1 #no gloss bsz if not training gloss encoder, memory concerns

	#create passed in ckpt dir if doesn't exist
	if not os.path.exists(args.ckpt): os.mkdir(args.ckpt)

	#set up tensorboard
	# NEW in FEWS: using tensorboard
	job_name = [a for a in args.ckpt.split('/') if len(a) > 0][-1]
	tb_writer = SummaryWriter('runs/{}'.format(job_name))

	'''
	LOAD PRETRAINED TOKENIZER
	'''
	tokenizer = load_tokenizer(args.encoder_name)

	model = model_loading(args)  # loading BEM, possibly from an existing checkpoint if specified in args
	'''
	LOADING IN TRAINING AND EVAL DATA
	'''
	print('Loading data + preprocessing...')
	sys.stdout.flush()
	#NOTE: NO longer overriding dev set
	#eval=''
	#if args.dataset == 'wn':
	#	eval='train_subset' #override dev set and validate on subset of training data

	train_data, eval_data, train_gloss_dict, eval_gloss_dict = data_loading(args, tokenizer, process_train=True)
	#if eval == 'train_subset': generate_gold_file(args.ckpt, eval_data)

	#calculate number of training steps per epoch and total train steps
	epochs = args.epochs
	train_steps = 0
	for batch in train_data:
		num_glosses = sum([len(train_gloss_dict[k][2]) for k in batch[3]])
		batch_steps = (num_glosses//args.gloss_bsz)+1
		train_steps += batch_steps

	t_total = train_steps*epochs

	criterion = torch.nn.CrossEntropyLoss(reduction='none')

	#optimize + scheduler from pytorch_transformers package
	#this taken from pytorch_transformers finetuning code
	weight_decay = 0.0 #this could be a parameter
	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
	adam_epsilon = 1e-8
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=adam_epsilon)
	schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup, num_training_steps=t_total)

	'''
	TRAIN MODEL ON SEMCOR DATA
	'''

	best_dev_f1 = 0.
	print('Training biencoder...')
	sys.stdout.flush()

	for epoch in range(1, epochs+1):
		#train model for one epoch or given number of training steps
		model, optimizer, schedule, train_loss = _train(train_data, model, train_gloss_dict, optimizer, schedule, criterion, gloss_bsz=args.gloss_bsz, max_grad_norm=args.grad_norm, silent=args.silent, multigpu=args.multigpu)

		#eval model on dev set (semeval2015)
		eval_preds = _eval(eval_data, model, eval_gloss_dict, multigpu=args.multigpu)

		if args.dataset == 'fews':
			eval_preds = [l for _, l in eval_preds]
			eval_labels = []
			for _, _, _, _, _, l in eval_data: eval_labels.extend(l)
			assert len(eval_preds) == len(eval_labels)
			dev_f1 = score(eval_preds, eval_labels)  # NEW in FEWS: looks like accuracy, not f1

		else:
			#generate predictions file
			pred_filepath = os.path.join(args.ckpt, 'tmp_predictions.txt')
			with open(pred_filepath, 'w') as f:
				for inst, prediction in eval_preds:
					f.write('{} {}\n'.format(inst, prediction))

			#run predictions through scorer
			if eval == 'train_subset': gold_filepath = os.path.join(args.ckpt, 'train_subset_gold.txt')
			else: gold_filepath = os.path.join(WN_DATAPATH, 'Evaluation_Datasets/semeval2015/semeval2015.gold.key.txt')
			scorer_path = os.path.join(WN_DATAPATH, 'Evaluation_Datasets')
			_, _, dev_f1 = evaluate_output(scorer_path, gold_filepath, pred_filepath)
			print('Dev f1 after {} epochs = {}'.format(epoch, dev_f1))
			sys.stdout.flush() 

		if dev_f1 >= best_dev_f1:
			print('updating best model at epoch {}...'.format(epoch))
			sys.stdout.flush() 
			best_dev_f1 = dev_f1
			#save to file if best probe so far on dev set
			model_fname = os.path.join(args.ckpt, 'best_model.ckpt')
			with open(model_fname, 'wb') as f:
				torch.save(model.state_dict(), f)
			sys.stdout.flush()

		#track perfromance on tensorboard
		tb_writer.add_scalar('dev_f1', dev_f1, epoch)

		#shuffle train set ordering after every epoch
		random.shuffle(train_data)
	return

def evaluate_model(args):
	# NEW in FEWS: supports FEWS now, calculates accuracy for FEWS and F1 for Raganato's
	if args.dataset == 'fews': split = args.fews_split
	else: split = args.wn_split
	print('Evaluating WSD model for {} on {}...'.format(args.dataset, split))

	'''
	LOAD TRAINED MODEL
	'''
	model = model_loading(args, eval_mode=True)  # loading BEM
	'''
	LOAD TOKENIZER
	'''
	tokenizer = load_tokenizer(args.encoder_name)

	'''
	LOAD EVAL SET
	'''
	_, eval_data, _, gloss_dict = data_loading(args, tokenizer, eval=split, process_train=False)

	'''
	EVALUATE MODEL
	'''
	eval_preds = _eval(eval_data, model, gloss_dict, multigpu=False)

	
	if args.dataset == 'wn':
		#generate predictions file
		pred_filepath = os.path.join(args.ckpt, './{}_predictions.txt'.format(split))
		with open(pred_filepath, 'w') as f:
			for inst, prediction in eval_preds:
				f.write('{} {}\n'.format(inst, prediction))

		#run predictions through scorer
		gold_filepath = os.path.join(WN_DATAPATH, 'Evaluation_Datasets/{}/{}.gold.key.txt'.format(split, split))
		scorer_path = os.path.join(WN_DATAPATH, 'Evaluation_Datasets')
		p, r, f1 = evaluate_output(scorer_path, gold_filepath, pred_filepath)
		print('f1 of WSD model on {} test set = {}'.format(split, f1))
	else: #fews scoring
		eval_preds = [l for _, l in eval_preds]
		eval_labels = []
		for _, _, _, _, _, l in eval_data: eval_labels.extend(l)
		acc = score(eval_preds, eval_labels)
		print('Acc of WSD model on {} set = {}'.format(split, acc))
		
		if args.fews_split in ['dev', 'test', 'dev-human-subset']:
			#in dev/test sets: first half = few-shot, second half = zero-shot
			split_idx = len(eval_labels)//2

			few_f1 = score(eval_preds[:split_idx], eval_labels[:split_idx])  # this is accuracy, not f1!
			zero_f1 = score(eval_preds[split_idx:], eval_labels[split_idx:])

			print('Few-shot acc = {}'.format(few_f1))
			print('Zero-shot acc = {}'.format(zero_f1))
		return 


	return

if __name__ == "__main__":
	#parse args
	args = parser.parse_args()
	print(args)

	if not torch.cuda.is_available() and args.device!='cpu':
		print("Need available GPU(s) to run this model or specify cpu argument...")
		quit()
	if args.device=='cpu':
		context_device = gloss_device = 'cpu'
	elif not args.multigpu:
		gloss_device = context_device


	#set random seeds
	torch.manual_seed(args.rand_seed)
	os.environ['PYTHONHASHSEED'] = str(args.rand_seed)
	torch.cuda.manual_seed(args.rand_seed)
	torch.cuda.manual_seed_all(args.rand_seed)   
	np.random.seed(args.rand_seed)
	random.seed(args.rand_seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic=True

	#evaluate model saved at checkpoint or...
	if args.eval: evaluate_model(args)
	#train model
	# NEW in FEWS: train_model_mtl() is called, but not defined
	# elif args.dataset == 'both': train_model_mtl(args)
	else: train_model(args)

#EOF
