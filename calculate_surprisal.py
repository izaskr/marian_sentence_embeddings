
import csv
from utils import surprisal_per_word_from_scratch
import pandas as pd
from transformers import AutoTokenizer, AutoModelWithLMHead

# read in data from .csv in surprisal/data/
path_data = "surprisal/data/data_in.csv"


# load the English GPT2 model
device = "cpu"
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
lm_model = AutoModelWithLMHead.from_pretrained(model_id).to(device)


all_ids, all_words, all_surprisals, all_conditions, all_target_regions = [], [], [], [], []
all_windices = []

with open(path_data, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter="\t")
    for j, line in enumerate(csv_reader):
        if j == 0:
            continue
        item_id = line[0]

        conditions = ["a_exp", "a_imp", "b_exp", "b_imp"]

        target_region_text = line[-1]
        # loop over the conditions of the item
        for k, text in enumerate(line[1:5]):
            _c = conditions[k]
            surprisals, subwords, corrected_tokens = surprisal_per_word_from_scratch(sentence=text, correcting=True,
                                                                 device=device,
                                                                 model_id=model_id, tokenizer=tokenizer, model=lm_model)
            target_region_repeat = [target_region_text] * len(surprisals)
            item_id_repeat, condition_repeat = [item_id] * len(surprisals), [_c] * len(surprisals)
            word_index = [w for w in range(1, len(surprisals)+1)]

            # populate the all_* lists to later export them as a csv from pandas
            all_ids.extend(item_id_repeat)
            all_conditions.extend(condition_repeat)
            all_target_regions.extend(target_region_repeat)
            all_words.extend(corrected_tokens)
            all_surprisals.extend(surprisals)
            all_windices.extend(word_index)


# pandas df
df = pd.DataFrame(list(zip(all_ids, all_conditions, all_words, all_windices, all_surprisals, all_target_regions)),
                  columns=["ItemID", "Condition", "Word", "WordID", "SurprisalGPT2", "TargetRegion"])

df.to_csv("surprisal/data/data_out.csv", index=False, encoding='utf-8')
