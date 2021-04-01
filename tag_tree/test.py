import pickle

with open("tag_tree/data/roberta/tagop_roberta_cached_train.pkl", 'rb') as f:
    data = pickle.load(f)

idex = 0
total = 0
for item in data:
    if item["operator_label"] <= 1:
        total += 1
        if item["span_pos_labels"][0][0] == -1:
            print("-------------------------------")
            print("index :{}".format(idex))

            print("span_pos_label: begin: {} end: {}".format(item["span_pos_labels"][0][0],
                                                             item["span_pos_labels"][0][1]))
            print(item["answer_dict"])

            idex += 1
print("total: %d" % total)