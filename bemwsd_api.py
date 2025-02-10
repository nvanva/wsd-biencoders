import hydra
from omegaconf import DictConfig

import biencoder
from biencoder import model_loading, _eval
from wsd_models.util import load_tokenizer
import torch
from tqdm import tqdm

class BEMWSD_Opt:
    def __init__(self, args):
        self.model = model_loading(args, eval_mode=True)  # loading BEM
        self.tokenizer = load_tokenizer(args.encoder_name)
        self.args = args
        self.context_cache, self.gloss_cache = {}, {}

    def vectorized_contexts(self, contexts, context_device):
        for sent, label in tqdm(contexts):
            if sent not in self.context_cache:
                print('Context missing in cache:', sent)
                self.context_cache[sent] = []
                eval_data = biencoder.preprocess_fews_context(self.tokenizer, [(sent, label)],
                                                              bsz=1, max_len=self.args.context_max_length)
                for context_ids, context_attn_mask, context_output_mask, example_keys, insts, _ in eval_data:
                    context_ids = context_ids.to(context_device)
                    context_attn_mask = context_attn_mask.to(context_device)
                    context_output = self.model.context_forward(context_ids, context_attn_mask, context_output_mask)
                    # TODO: when many examples it makes sense to move context_output to CPU and free GPU memory
                    for output, key, inst in zip(context_output.split(1, dim=0), example_keys, insts):
                        self.context_cache[sent].append((output, key, inst))

            for output, key, inst in self.context_cache[sent]:
                yield output, key, inst


    # def vectorized_glosses(self, gloss_dict, key, gloss_device):
    #     gloss_ids, gloss_attn_mask, sense_keys = gloss_dict[key]
    #     gloss_ids = gloss_ids.to(gloss_device)
    #     gloss_attn_mask = gloss_attn_mask.to(gloss_device)
    #     gloss_output = self.model.gloss_forward(gloss_ids, gloss_attn_mask)
    #     return gloss_output, sense_keys

    def vectorized_glosses(self, key, inventory, gloss_device):
        tocache = [(sid, gloss) for sid, gloss in zip(*inventory[key]) if gloss not in self.gloss_cache]
        if tocache:
            sids, glosses = zip(*tocache)
            sids, glosses = list(sids), list(glosses)
            print(f'Glosses missing in cache: for key {key} {len(glosses)}/{len(inventory[key][0])} glosses are missing')
            data = [('', sids[0])]
            gloss_dict = biencoder.preprocess_fews_glosses(data, self.tokenizer, {key: (sids, glosses)},
                                                           max_len=self.args.gloss_max_length)

            gloss_ids, gloss_attn_mask, sense_keys = gloss_dict[key]
            gloss_ids = gloss_ids.to(gloss_device)
            gloss_attn_mask = gloss_attn_mask.to(gloss_device)
            gloss_output = self.model.gloss_forward(gloss_ids, gloss_attn_mask)
            # TODO: probably move gloss_output to cpu
            for gloss, output in zip(glosses, gloss_output):
                self.gloss_cache[gloss] = output

        sids, glosses = inventory[key]
        gloss_output = torch.concat([self.gloss_cache[gloss].reshape(1,-1) for gloss in glosses])
        return gloss_output, sids


    def wsd(self, usages, inventory):
        """
        For each usage from a list of usages find the most suitable definition in the given inventory.
        :param usages: a list of (text, start, end, inventory key)
        :param inventory: a dictionary from inventory keys (e.g. lemma.pos) to a tuple (list[sids], list[definitions]) to select from
        :return: a list of selected sids parallel to the input list of usages
        """
        assert all(len(sids)==len(glosses) for sids, glosses in inventory.values())
        # the original BEM code we call requires specific naming of sids and their correspondence to keys, so we rename both
        iitems = list(inventory.items())  # order is important to build the oldkey->newkey mapping
        oldkey2newkey = {skey: f'{i}.x' for i, (skey, _) in enumerate(iitems)}
        inventory = {f'{i}.x': ([f'{i}.x.{s}' for s in sids], glosses)  # prepending 2 dot-separated parts matching the new keys[
                     for  i, (skey, (sids, glosses)) in enumerate(iitems)}

        contexts = []
        for s, st, en, oldkey in usages:
            ctx = f'{s[:st]}<WSD>{s[st:en]}</WSD>{s[en:]}'
            label = inventory[oldkey2newkey[oldkey]][0][0]  # can use any sid as the gold label because no evaluation, just use the first
            contexts.append((ctx,label))

        gloss_dict = biencoder.preprocess_fews_glosses(contexts, self.tokenizer, inventory,
                                                       max_len=self.args.gloss_max_length)

        eval_preds = []
        self.model.eval()
        with torch.no_grad():
            for output, key, inst in self.vectorized_contexts(contexts, self.args.device):
                # gloss_output, sense_keys = self.vectorized_glosses(gloss_dict, key, self.args.device)
                gloss_output, sense_keys = self.vectorized_glosses(key, inventory, self.args.device)
                gloss_output = gloss_output.transpose(0, 1)

                # get cosine sim of example from context encoder with gloss embeddings
                output = torch.mm(output, gloss_output)
                pred_idx = output.topk(1, dim=-1)[1].squeeze().item()
                pred_label = sense_keys[pred_idx]
                eval_preds.append((inst, pred_label))

        res = ['.'.join(pred.split('.')[2:]) for inst, pred in eval_preds]  # convert back to the original class names
        return res


class BEMWSD:
    def __init__(self, args):
        self.model = model_loading(args, eval_mode=True)  # loading BEM
        self.tokenizer = load_tokenizer(args.encoder_name)
        self.args = args


    def wsd(self, usages, inventory):
        """
        For each usage from a list of usages find the most suitable definition in the given inventory.
        :param usages: a list of (text, start, end, inventory key)
        :param inventory: a dictionary from inventory keys (e.g. lemma.pos) to a tuple (list[sids], list[definitions]) to select from
        :return: a list of selected sids parallel to the input list of usages
        """
        assert all(len(sids)==len(glosses) for sids, glosses in inventory.values())
        # the original BEM code we call requires specific naming of sids and their correspondence to keys, so we rename both
        iitems = list(inventory.items())  # order is important to build the oldkey->newkey mapping
        oldkey2newkey = {skey: f'{i}.x' for i, (skey, _) in enumerate(iitems)}
        inventory = {f'{i}.x': ([f'{i}.x.{s}' for s in sids], glosses)  # prepending 2 dot-separated parts matching the new keys[
                     for  i, (skey, (sids, glosses)) in enumerate(iitems)}

        contexts = []
        for s, st, en, oldkey in usages:
            ctx = f'{s[:st]}<WSD>{s[st:en]}</WSD>{s[en:]}'
            label = inventory[oldkey2newkey[oldkey]][0][0]  # can use any sid as the gold label because no evaluation, just use the first
            contexts.append((ctx,label))
        gloss_dict = biencoder.preprocess_fews_glosses(contexts, self.tokenizer, inventory, max_len=self.args.gloss_max_length)
        eval_data = biencoder.preprocess_fews_context(self.tokenizer, contexts, bsz=1, max_len=self.args.context_max_length)
        eval_preds = _eval(eval_data, self.model, gloss_dict, self.args.device)
        res = ['.'.join(pred.split('.')[2:]) for inst, pred in eval_preds]  # convert back to the original class names
        return res


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    bemwsd = BEMWSD(cfg)

    texts = [
        'I went to the ***bank*** of the river.',
        '***Banks*** have lots of money.',
        '***Bank*** is ambiguous, it can be a financial ***bank*** or a ***bank*** of a river of other object with water.',
        # for the last we will do wsd only for the first occurrence of bank (though BEM supports multiple instances
        # in a single context our API requires one start/end per usage for simplicity)
    ]

    usages = []
    for t in texts:
        ff = t.split('***')
        usages.append( (''.join(ff), len(ff[0]), len(ff[0]) + len(ff[1]), 'bank.noun') )

    inventory = {'bank.noun': (['bank.noun.1','bank.noun.2'],['a financial organization','a slope of ground near the water'])}
    eval_preds = bemwsd.wsd(usages, inventory)
    print(eval_preds)


if __name__ == "__main__":
    main()

