import json

with open("tatqa_dataset_dev.json") as f:
    data = json.load(f)
index = 0
for one in data:
    questions = one['questions']
    index += 1
    for question_answer in questions:
        if question_answer["facts"][0] == 'See Note 7 of Notes to Consolidated Financial Statements included elsewhere in this Annual report for additional information regarding our notes payable and other borrowings.':
            with open('new.json', 'w') as f:
               json.dump(one,f)
            print(index)
            break
