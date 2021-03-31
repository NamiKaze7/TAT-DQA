import pickle

with open("tag_op/data/roberta/tagop_roberta_cached_train.pkl", 'rb') as f:
    data = pickle.load(f)

idex = 0

for item in data:
    if item["operator_label"] <= 1:
       if item["span_pos_labels"][0] == -1:
           print(item["answer_dict"])

