# -*- coding:utf-8 -*-
# this code is used to create a pretrained word2vec
import json
use_names = ['dfa', 'dfb', 'proxy']
data_dir = 'data_s2s/'
splits = ['training', 'dev', 'test']


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
        out_file_enc = open('data_s2s/word2vec_enc_corpus.txt', 'w', encoding='UTF-8')
        out_file_dec = open('data_s2s/word2vec_dec_corpus.txt', 'w', encoding='UTF-8')
        for split in splits:
            data = json.load(open(data_dir + 'all-' + split + '.json', 'r', encoding='UTF-8'))
            for d in data:
                snt = '<s> ' + ' '.join(d['sent'].split('<SPACE>')) + ' </s>'
                snt_pos = '<s> ' + ' '.join(d['sent_mask'].split('<SPACE>')) + ' </s>'
                amr = d['amr']
                out_file_dec.write(snt + '\n')
                out_file_dec.write(snt_pos + '\n')
                out_file_enc.write(amr + '\n')  # ???
        out_file_enc.close()
        out_file_dec.close()

    def edit_wordvec():
        vec_names = ['enc0.vec', 'dec0.vec', 'enc1.vec', 'enc2.vec','dec1.vec', 'dec2.vec']
        for vec_name in vec_names:
            in_file = open('data_s2s/' + vec_name, 'r', encoding='UTF-8').read().strip().split('\n')
            out_file = open('data_s2s/' + 'FastText_' + vec_name, 'w', encoding='UTF-8')
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

    # create_wordvec_corpus()
    # then use the FastText to train own vec
    edit_wordvec()


def bpevec_pipeline():
    def create_bpe_corpus():
        out_file_enc = open('data_s2s/bpe_enc.corpus', 'w', encoding='UTF-8')
        out_file_dec = open('data_s2s/bpe_dec.corpus', 'w', encoding='UTF-8')
        for split in splits:
            data = json.load(open(data_dir + 'all-' + split + '.json', 'r', encoding='UTF-8'))
            split_file_enc = open('data_s2s/bpe-all-enc-' + split + '.corpus', 'w', encoding='UTF-8')
            split_file_dec = open('data_s2s/bpe-all-dec-' + split + '.corpus', 'w', encoding='UTF-8')
            for d in data:
                snt = ' '.join(d['sent'].split('<SPACE>'))
                snt_pos = ' '.join(d['sent_mask'].split('<SPACE>'))
                amr = d['amr']
                split_file_dec.write(snt + '\n')
                split_file_dec.write(snt_pos + '\n')
                split_file_enc.write(amr + '\n')
                out_file_dec.write(snt + '\n')
                out_file_dec.write(snt_pos + '\n')
                out_file_enc.write(amr + '\n')
            split_file_enc.close()
            split_file_dec.close()
        out_file_enc.close()
        out_file_dec.close()

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
            ori_data = json.load(open('data_s2s/all-' + split + '.json', 'r', encoding='UTF-8'))
            bpe_data_enc = open('data_s2s/all-enc-' + split + '-out.bpe' + str(bpe_num), 'r', encoding='UTF-8').read().strip().split('\n')
            bpe_data_dec = open('data_s2s/all-dec-' + split + '-out.bpe' + str(bpe_num), 'r', encoding='UTF-8').read().strip().split('\n')
            for i, data in enumerate(ori_data):
                data['sent_bpe'] = bpe_data_dec[i * 2]
                data['sent_mask_bpe'] = manage_sent_mask_bpe(bpe_data_dec[i * 2 + 1])  # concat the split mask token
                data['amr_bpe'] = bpe_data_enc[i]
            json.dump(ori_data, open('data_s2s/all-' + split + '-bpe' + str(bpe_num) + '.json', 'w', encoding='UTF-8'))

    def create_bpevec_corpus(bpe_num):
        out_file_enc = open('data_s2s/bpe' + str(bpe_num) +'_2vec_enc_corpus.txt', 'w', encoding='UTF-8')
        out_file_dec = open('data_s2s/bpe' + str(bpe_num) +'_2vec_dec_corpus.txt', 'w', encoding='UTF-8')
        for split in splits:
            data = json.load(open(data_dir + 'all-' + split + '-bpe' + str(bpe_num) + '.json', 'r', encoding='UTF-8'))
            for i, d in enumerate(data):
                snt = '<s> ' + d['sent_bpe'] + ' </s>'
                snt_pos = '<s> ' + d['sent_mask_bpe'] + ' </s>'
                amr = d['amr_bpe']
                out_file_dec.write(snt + '\n')
                out_file_dec.write(snt_pos + '\n')
                out_file_enc.write(amr + '\n')
        out_file_dec.close()
        out_file_enc.close()

    def edit_bpevec():
        vec_names = ['bpe5000.vec', 'bpe6000.vec', 'bpe7000.vec','bpe8000.vec']  # 'bpe4000.vec', 'bpe10000.vec', 'bpe12000.vec', 'bpe15000.vec','bpe20000.vec'
        for vec_name in vec_names:
            in_file_enc = open('data_s2s/enc_' + vec_name, 'r', encoding='UTF-8').read().strip().split('\n')
            in_file_dec = open('data_s2s/dec_' + vec_name, 'r', encoding='UTF-8').read().strip().split('\n')
            out_file_enc = open('data_s2s/' + 'FastText_enc_' + vec_name, 'w', encoding='UTF-8')
            out_file_dec = open('data_s2s/' + 'FastText_dec_' + vec_name, 'w', encoding='UTF-8')
            for i, line in enumerate(in_file_enc):
                line = line.strip().split(' ')
                if i == 0:
                    word_num = int(line[0])
                    vec_dim = int(line[1])
                    print('word num is %d, word vec dim is %d\n' % (word_num, vec_dim))
                    pad_vec = ['0.000000'] * vec_dim
                    out_file_enc.write('%d\t#pad#\t%s\n' % (i, ' '.join(pad_vec)))
                    continue
                out_file_enc.write('%d\t%s\t%s\n' % (i, line[0], ' '.join(line[1:])))

            for i, line in enumerate(in_file_dec):
                line = line.strip().split(' ')
                if i == 0:
                    word_num = int(line[0])
                    vec_dim = int(line[1])
                    print('word num is %d, word vec dim is %d\n' % (word_num, vec_dim))
                    pad_vec = ['0.000000'] * vec_dim
                    out_file_dec.write('%d\t#pad#\t%s\n' % (i, ' '.join(pad_vec)))
                    continue
                out_file_dec.write('%d\t%s\t%s\n' % (i, line[0], ' '.join(line[1:])))


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
    # # then use the subword nmt to train a bpe-codes and get name-split-out.bpe*
    # bpe_nums = [5000, 6000, 7000, 8000]  # 4000, 10000, 12000, 15000, 20000
    # for bpe_num in bpe_nums:
    #     print("bpe_num: %d" % bpe_num)
    #     AddBpeData2json(bpe_num)
    #     create_bpevec_corpus(bpe_num)
    #     check_sent_mask_bpe(bpe_num)
    # # then use the FastText to train own vec
    # edit_bpevec()


wordvec_pipeline()
# bpevec_pipeline()

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

