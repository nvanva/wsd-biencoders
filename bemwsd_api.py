import hydra
from omegaconf import DictConfig

import biencoder
from biencoder import model_loading, _eval
from wsd_models.util import load_tokenizer


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
        contexts = []
        for s, st, en, key in usages:
            ctx = f'{s[:st]}<WSD>{s[st:en]}</WSD>{s[en:]}'
            label = inventory[key][0][0]  # can use any sid as the gold label because no evaluation, just use the first
            contexts.append((ctx,label))
        gloss_dict = biencoder.preprocess_fews_glosses(contexts, self.tokenizer, inventory, max_len=self.args.gloss_max_length)
        eval_data = biencoder.preprocess_fews_context(self.tokenizer, contexts, bsz=1, max_len=self.args.context_max_length)
        eval_preds = _eval(eval_data, self.model, gloss_dict, multigpu=False)
        res = [pred for inst, pred in eval_preds]
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

