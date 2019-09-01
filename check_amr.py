import json
path = 'data/proxy-training.json'
data = json.load(open(path, 'r'))
for a in data:
    amr_lst = a['amr']
    id = a['id']
    assert sum(x == '(' for x in amr_lst) == sum(x == ')' for x in amr_lst), '{0} {1}'.format(amr_lst, id)

