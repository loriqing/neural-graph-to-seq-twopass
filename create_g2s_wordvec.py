# -*- coding:utf-8 -*-
# this code is used to create a pretrained word2vec
import json
use_names = ['dfa', 'dfb', 'proxy']
data_dir = 'data/'
splits = ['training', 'dev', 'test']

def read_anonymized(amr_lst, amr_node, amr_edge):
    assert sum(x=='(' for x in amr_lst) == sum(x==')' for x in amr_lst), '{0}'.format(amr_lst)
    cur_str = amr_lst[0]
    cur_id = len(amr_node)
    amr_node.append(cur_str)
    # print(amr_lst)
    i = 1
    while i < len(amr_lst):
        if amr_lst[i].startswith(':') == False: ## cur cur-num_0
            nxt_str = amr_lst[i]
            nxt_id = len(amr_node)
            amr_node.append(nxt_str)
            amr_edge.append((cur_id, nxt_id, ':value'))
            i = i + 1
        elif amr_lst[i].startswith(':') and len(amr_lst) == 2: ## cur :edge
            nxt_str = 'num_unk'
            nxt_id = len(amr_node)
            amr_node.append(nxt_str)
            amr_edge.append((cur_id, nxt_id, amr_lst[i]))
            i = i + 1
        elif amr_lst[i].startswith(':') and amr_lst[i+1] != '(': ## cur :edge nxt
            nxt_str = amr_lst[i+1]
            nxt_id = len(amr_node)
            amr_node.append(nxt_str)
            amr_edge.append((cur_id, nxt_id, amr_lst[i]))
            i = i + 2
        elif amr_lst[i].startswith(':') and amr_lst[i+1] == '(': ## cur :edge ( ... )
            number = 1
            j = i+2
            while j < len(amr_lst):
                number += (amr_lst[j] == '(')
                number -= (amr_lst[j] == ')')
                if number == 0:
                    break
                j += 1
            assert number == 0 and amr_lst[j] == ')', ' '.join(amr_lst[i+2:j])
            nxt_id = read_anonymized(amr_lst[i+2:j], amr_node, amr_edge)
            amr_edge.append((cur_id, nxt_id, amr_lst[i]))
            i = j + 1
        else:
            assert False, ' '.join(amr_lst)
        # print(amr_node)
        # print(amr_edge)

    return cur_id


def read_bpe_anonymized(amr_lst, amr_node, amr_edge):  # add a new type :bpe
    assert sum(x=='(' for x in amr_lst) == sum(x==')' for x in amr_lst), '{0}'.format(amr_lst)
    # print(' '.join(amr_lst))
    cur_str = amr_lst[0]
    cur_id = len(amr_node)
    temp_cur_id = -1
    if cur_str[-2:] == '@@':
        temp_cur_id = len(amr_node)  # the first bpe word
    amr_node.append(cur_str)
    i = 1
    while i < len(amr_lst):
        if amr_lst[i].startswith(':') == False and amr_lst[i-1][-2:] == '@@': ## suffix of the node
            nxt_str = amr_lst[i]
            nxt_id = len(amr_node)
            amr_node.append(nxt_str)
            amr_edge.append((temp_cur_id, nxt_id, ':bpe'))
            temp_cur_id = nxt_id
            i = i + 1
        elif amr_lst[i].startswith(':') == False: ## cur cur-num_0
            nxt_str = amr_lst[i]
            nxt_id = len(amr_node)
            amr_node.append(nxt_str)
            amr_edge.append((cur_id, nxt_id, ':value'))
            if amr_lst[i][-2:] == '@@':
                temp_cur_id = nxt_id
            i = i + 1
        elif amr_lst[i].startswith(':'):
            edge = amr_lst[i]
            if edge[-2:] == '@@':  ## suffix of the edge
                while edge[-2:] == '@@':
                    i = i + 1
                    edge = edge[:-2] + amr_lst[i]
            if i+1 == len(amr_lst):
                nxt_str = 'num_unk'
                nxt_id = len(amr_node)
                amr_node.append(nxt_str)
                amr_edge.append((cur_id, nxt_id, edge))
                i = i + 1
            elif amr_lst[i+1] != '(':
                nxt_str = amr_lst[i + 1]
                nxt_id = len(amr_node)
                amr_node.append(nxt_str)
                amr_edge.append((cur_id, nxt_id, edge))
                if nxt_str[-2:] == '@@':
                    temp_cur_id = nxt_id
                i = i + 2
            elif amr_lst[i+1] == '(':
                number = 1
                j = i + 2
                while j < len(amr_lst):
                    number += (amr_lst[j] == '(')
                    number -= (amr_lst[j] == ')')
                    if number == 0:
                        break
                    j += 1
                assert number == 0 and amr_lst[j] == ')', ' '.join(amr_lst[i + 2:j])
                nxt_id = read_bpe_anonymized(amr_lst[i + 2:j], amr_node, amr_edge)
                amr_edge.append((cur_id, nxt_id, edge))
                i = j + 1
            else:
                assert False, ' '.join(amr_lst)
        else:
            assert False, ' '.join(amr_lst)
        # print(amr_node)
        # print(amr_edge)

    return cur_id


def merge_all_data():
    for split in splits:
        all_data = []
        for use_name in use_names:
            in_file = open(data_dir + use_name + '-' + split + '.json', 'r', encoding='UTF-8')
            data = json.load(in_file)
            all_data += data
            in_file.close()
        out_file = open(data_dir + 'all-' + split + '.json', 'w', encoding='UTF-8')
        json.dump(all_data, out_file)
        out_file.close()


def wordvec_pipeline():
    def create_wordvec_corpus():
        out_file = open('data/word2vec_corpus1.txt', 'w', encoding='UTF-8')
        for split in splits:
            data = json.load(open(data_dir + 'all-' + split + '.json', 'r', encoding='UTF-8'))
            for d in data:
                snt = '<s> ' + ' '.join(d['sent'].split('<SPACE>')) + ' </s>'
                snt_pos = '<s> ' + ' '.join(d['sent_mask'].split('<SPACE>')) + ' </s>'
                amr = d['amr']
                amr_node = []
                amr_edge = []
                read_anonymized(amr.strip().split(), amr_node, amr_edge)
                out_file.write(snt + '\n')
                out_file.write(snt_pos + '\n')
                out_file.write(' '.join(amr_node) + '\n')  # ???
        out_file.close()

    def edit_wordvec():
        vec_names = ['node_1.vec', 'node_2.vec', 'no_node_1.vec', 'no_node_2.vec']
        for vec_name in vec_names:
            in_file = open('data/' + vec_name, 'r', encoding='UTF-8').read().strip().split('\n')
            out_file = open('data/' + 'FastText_' + vec_name, 'w', encoding='UTF-8')
            for i, line in enumerate(in_file):
                line = line.strip().split(' ')
                if i == 0:
                    word_num = int(line[0])
                    vec_dim = int(line[1])
                    print('word num is %d, word vec dim is %d\n' % (word_num, vec_dim))
                    pad_vec = ['0.000000'] * vec_dim
                    out_file.write('%d\t#pad#\t%s\n' % (i, ' '.join(pad_vec)))
                    continue
                out_file.write('%d\t%s\t%s\n' % (i, line[0], ' '.join(line[1:])))
            out_file.close()

    create_wordvec_corpus()
    # then use the FastText to train own vec
    edit_wordvec()


def bpevec_pipeline():
    def create_bpe_corpus():
        out_file = open('data/bpe.corpus', 'w', encoding='UTF-8')
        for split in splits:
            data = json.load(open(data_dir + 'all-' + split + '.json', 'r', encoding='UTF-8'))
            split_file = open('data/all-' + split + '-bpe.corpus', 'w', encoding='UTF-8')
            for d in data:
                snt = ' '.join(d['sent'].split('<SPACE>'))
                snt_pos = ' '.join(d['sent_mask'].split('<SPACE>'))
                amr = d['amr']
                split_file.write(snt + '\n')
                split_file.write(snt_pos + '\n')
                split_file.write(amr + '\n')
                out_file.write(snt + '\n')
                out_file.write(snt_pos + '\n')
                out_file.write(amr + '\n')
            split_file.close()
        out_file.close()

    def AddBpeData2json(bpe_num):
        def manage_sent_mask_bpe(snt_pos):  # concat the split mask token
            snt_pos = snt_pos.strip().split()
            edit_snt = snt_pos[0]
            i = 1
            while i < len(snt_pos):
                if snt_pos[i][0]>='A' and snt_pos[i][0]<='Z' and snt_pos[i-1][-2:]=='@@':
                    edit_snt = edit_snt[:-2] + snt_pos[i]
                else:
                    edit_snt = edit_snt + ' ' + snt_pos[i]
                i += 1
            return edit_snt
        for split in splits:
            ori_data = json.load(open('data/all-' + split + '.json', 'r', encoding='UTF-8'))
            bpe_data = open('data/all-' + split + '-out.bpe' + str(bpe_num), 'r', encoding='UTF-8').read().strip().split('\n')
            for i, data in enumerate(ori_data):
                data['sent_bpe'] = bpe_data[i * 3]
                data['sent_mask_bpe'] = manage_sent_mask_bpe(bpe_data[i * 3 + 1])  # concat the split mask token
                data['amr_bpe'] = bpe_data[i * 3 + 2]
            json.dump(ori_data, open('data/all-' + split + '-bpe' + str(bpe_num) + '.json', 'w', encoding='UTF-8'))

    def create_bpevec_corpus(bpe_num):
        out_file = open('data/bpe' + str(bpe_num) +'_2vec_corpus.txt', 'w', encoding='UTF-8')
        for split in splits:
            data = json.load(open(data_dir + 'all-' + split + '-bpe' + str(bpe_num) + '.json', 'r', encoding='UTF-8'))
            for i, d in enumerate(data):
                snt = '<s> ' + d['sent_bpe'] + ' </s>'
                snt_pos = '<s> ' + d['sent_mask_bpe'] + ' </s>'
                amr = d['amr_bpe']
                amr_node = []
                amr_edge = []
                try:
                    read_bpe_anonymized(amr.strip().split(), amr_node, amr_edge)
                except:
                    print('error!')
                    print(i)
                    print(d['amr'])
                    print(d['amr_bpe'])
                    print(amr_node)
                    print(amr_edge)
                    exit(0)
                out_file.write(snt + '\n')
                out_file.write(snt_pos + '\n')
                out_file.write(' '.join(amr_node) + '\n')
        out_file.close()

    def edit_bpevec():
        vec_names = ['bpe4000.vec']  # ['bpe5000.vec', 'bpe6000.vec', 'bpe7000.vec','bpe8000.vec','bpe10000.vec', 'bpe12000.vec', 'bpe15000.vec','bpe20000.vec']
        for vec_name in vec_names:
            in_file = open('data/' + vec_name, 'r', encoding='UTF-8').read().strip().split('\n')
            out_file = open('data/' + 'FastText_' + vec_name, 'w', encoding='UTF-8')
            for i, line in enumerate(in_file):
                line = line.strip().split(' ')
                if i == 0:
                    word_num = int(line[0])
                    vec_dim = int(line[1])
                    print('word num is %d, word vec dim is %d\n' % (word_num, vec_dim))
                    pad_vec = ['0.000000'] * vec_dim
                    out_file.write('%d\t#pad#\t%s\n' % (i, ' '.join(pad_vec)))
                    continue
                out_file.write('%d\t%s\t%s\n' % (i, line[0], ' '.join(line[1:])))
            out_file.close()

    def check_sent_mask_bpe(bpe_num):  # concat the split mask token
        def manage_sent_mask_bpe(snt_pos):  # concat the split mask token
            snt_pos = snt_pos.strip().split()
            edit_snt = snt_pos[0]
            i = 1
            while i < len(snt_pos):
                if snt_pos[i][0] >= 'A' and snt_pos[i][0] <= 'Z' and snt_pos[i - 1][-2:] == '@@':
                    edit_snt = edit_snt[:-2] + snt_pos[i]
                    print(snt_pos[i])
                i += 1
        for split in splits:
            data = json.load(open(data_dir + 'all-' + split + '-bpe' + str(bpe_num) + '.json', 'r', encoding='UTF-8'))
            for d in data:
                manage_sent_mask_bpe(d['sent_mask_bpe'])
        print('check sent mask ok!')

    # create_bpe_corpus()
    # then use the subword nmt to train a bpe-codes and get name-split-out.bpe*
    # bpe_nums = [4000]  # ,5000, 6000, 7000, 8000, 10000, 12000, 15000, 20000
    # for bpe_num in bpe_nums:
    #     print("bpe_num: %d" % bpe_num)
    #     AddBpeData2json(bpe_num)
    #     create_bpevec_corpus(bpe_num)
    #     check_sent_mask_bpe(bpe_num)
    # then use the FastText to train own vec
    edit_bpevec()




# wordvec_pipeline()
bpevec_pipeline()

#### test
# amr1 = 'possible :polarity - :arg1 ( and :op1 ( feed :arg0 you :arg2 baby ) :op2 ( house :arg0 you :arg1 baby ) :op3 ( care :arg0 you :arg1 baby ) ) :condition-of ( have :polarity - :arg0 you :arg1 ( business :purpose ( have :arg0 you :arg1 baby ) ) )'
# amr2 = 'possible :polarity - :arg1 ( and :op1 ( feed :arg0 you :arg2 baby ) :op2 ( house :arg0 you :arg1 baby ) :op3 ( care :arg0 you :arg1 baby ) ) :con@@ di@@ tion-of ( have :polarity - :arg0 you :arg1 ( business :purpose ( have :arg0 you :arg1 baby ) ) )'
# amr_node = []
# amr_edge = []
# read_anonymized(amr1.strip().split(), amr_node, amr_edge)
# print(amr_node)
# print(amr_edge)
# amr_node = []
# amr_edge = []
# read_bpe_anonymized(amr2.strip().split(), amr_node, amr_edge)
# print(amr_node)
# print(amr_edge)
