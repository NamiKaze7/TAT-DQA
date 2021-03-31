import pickle

with open("tag_op/data/roberta/tagop_roberta_cached_train.pkl", 'rb') as f:
    data = pickle.load(f)

idex = 0

for item in data:
    print(item["span_pos_labels"][0])
    print('----------------------------')
    idex += 1
    if idex > 9:
        break
