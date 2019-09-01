# -*- coding:utf-8 -*-
# note: this code should run in python3 env
# create amr, id, snt, snt_space, pos_space files and split the domain without dev and test
import sys
import io
import os
import string
from stanfordcorenlp import StanfordCoreNLP
import re

nlp = StanfordCoreNLP(r'../../stanford-corenlp-full-2018-10-05/')

data_dir = 'data/amrs/split/'
middle_name = '/amr-release-2.0-amrs-'
use_names = ['dfa', 'dfb', 'proxy']
splits = ['training', 'dev', 'test']


def get_all_data():
    for split in splits:
        for name in use_names:
            id_dict = {}
            ids = []
            snts = []
            snts_space = []
            poses = []
            amrs = []
            amr_str = ''
            file_name = data_dir + split + middle_name + split + '-' + name + '.txt'
            if not os.path.exists(file_name):
                continue
            for line in open(file_name, 'r', encoding='UTF-8'):
                if line.startswith('#'):
                    if line.startswith('# ::id'):
                        id = line.lower().strip().split()[2]
                        ids.append(id)
                        id_dict[id] = len(ids)-1
                    if line.startswith('# ::snt'):
                        snt = line[8:]
                        pos_snt = nlp.pos_tag(snt)
                        words = []
                        pos = []
                        for w, tag in pos_snt:
                            words.append(w.lower().strip())
                            pos.append(tag)
                        snts.append(' '.join(words))
                        snts_space.append('<SPACE>'.join(words))
                        poses.append('<SPACE>'.join(pos))
                    continue
                line = line.strip()
                if line == '':
                    if amr_str != '':
                        if '~' in amr_str:
                            print(amr_str)
                            amr_str = amr_str.translate(str.maketrans('', '', '~'))
                        if '#' in amr_str:
                            print(amr_str)
                            amr_str = amr_str.replace('#', 'replacewikisharpe')
                        astr = amr_str
                        amr_str = ''
                        while '"' in astr:
                            amr_str += astr[:astr.find('"')+1]
                            astr = astr[astr.find('"') + 1:]
                            amr_str += astr[:astr.find('"')+1].replace(' ', 'replacespace')
                            astr = astr[astr.find('"') + 1:]
                        amr_str += astr

                        amrs.append(amr_str.strip())
                        amr_str = ''
                else:
                    amr_str = amr_str + line + ' '

            if amr_str != '':
                if '~' in amr_str:
                    print(amr_str)
                    amr_str = amr_str.translate(str.maketrans('', '', '~'))
                if '#' in amr_str:
                    print(amr_str)
                    amr_str = amr_str.replace('#', 'replacewikisharpe')
                astr = amr_str
                amr_str = ''
                while '"' in astr:
                    amr_str += astr[:astr.find('"')+1]
                    astr = astr[astr.find('"') + 1:]
                    amr_str += astr[:astr.find('"')+1].replace(' ', 'replacespace')
                    astr = astr[astr.find('"') + 1:]
                amr_str += astr

                amrs.append(amr_str.strip())
                amr_str = ''

            assert len(amrs) == len(ids) and len(snts) == len(poses) and len(amrs) == len(snts), '{0}, {1}, {2}, {3}'.format(len(amrs), len(ids), len(poses), len(snts))
            out_amr_file = 'data/' + name + '-' + split + '.amr'
            out_amr_f = open(out_amr_file, 'w')
            out_id_file = 'data/' + name + '-' + split + '.id'
            out_id_f = open(out_id_file, 'w')
            out_snt_file = 'data/' + name + '-' + split + '.snt'
            out_snt_f = open(out_snt_file, 'w')
            out_snt_space_file = 'data/' + name + '-' + split + '.snt_space'
            out_snt_space_f = open(out_snt_space_file, 'w')
            out_pos_file = 'data/' + name + '-' + split + '.pos'
            out_pos_f = open(out_pos_file, 'w')

            for idx in range(len(amrs)):
                amr = amrs[idx]
                id = ids[idx]
                snt = snts[idx]
                snt_space = snts_space[idx]
                pos = poses[idx]
                out_amr_f.write(amr + '\n')
                out_id_f.write(id + '\n')
                out_snt_f.write(snt + '\n')
                out_snt_space_f.write(snt_space + '\n')
                out_pos_f.write(pos + '\n')

            out_id_f.close()
            out_amr_f.close()
            out_snt_f.close()
            out_snt_space_f.close()
            out_pos_f.close()


def create_dev_test():
    for name in use_names:
        if os.path.exists(data_dir + 'dev' + middle_name + 'dev-' + name + '.txt'):
            continue

        amr_file = 'data/' + name + '-training' + '.amr'
        dev_amr_f = open('data/' + name + '-dev' + '.amr', 'w', encoding='UTF-8')
        test_amr_f = open('data/' + name + '-test' + '.amr', 'w', encoding='UTF-8')
        amrs = open(amr_file, 'r').read().strip().split('\n')

        id_file = 'data/' + name + '-training' + '.id'
        dev_id_f = open('data/' + name + '-dev' + '.id', 'w', encoding='UTF-8')
        test_id_f = open('data/' + name + '-test' + '.id', 'w', encoding='UTF-8')
        ids = open(id_file, 'r').read().strip().split('\n')

        snt_file = 'data/' + name + '-training' + '.snt'
        dev_snt_f = open('data/' + name + '-dev' + '.snt', 'w', encoding='UTF-8')
        test_snt_f = open('data/' + name + '-test' + '.snt', 'w', encoding='UTF-8')
        snts = open(snt_file, 'r').read().strip().split('\n')

        snt_space_file = 'data/' + name + '-training' + '.snt_space'
        dev_snt_space_f = open('data/' + name + '-dev' + '.snt_space', 'w', encoding='UTF-8')
        test_snt_space_f = open('data/' + name + '-test' + '.snt_space', 'w', encoding='UTF-8')
        snts_space = open(snt_space_file, 'r').read().strip().split('\n')

        pos_file = 'data/' + name + '-training' + '.pos'
        dev_pos_f = open('data/' + name + '-dev' + '.pos', 'w', encoding='UTF-8')
        test_pos_f = open('data/' + name + '-test' + '.pos', 'w', encoding='UTF-8')
        poses = open(pos_file, 'r').read().strip().split('\n')

        num = int(len(amrs)/10)
        gap = 5
        original_num = len(amrs)
        split = 0
        for idx in range(num)[::-1]:
            split += 1
            dev_amr_f.write(amrs[idx*10] + '\n')
            test_amr_f.write(amrs[idx*10+gap] + '\n')
            amrs.pop(idx * 10 + gap)
            amrs.pop(idx * 10)
            dev_id_f.write(ids[idx * 10] + '\n')
            test_id_f.write(ids[idx * 10 + gap] + '\n')
            ids.pop(idx * 10 + gap)
            ids.pop(idx * 10)
            dev_snt_f.write(snts[idx * 10] + '\n')
            test_snt_f.write(snts[idx * 10 + gap] + '\n')
            snts.pop(idx * 10 + gap)
            snts.pop(idx * 10)
            dev_snt_space_f.write(snts_space[idx * 10] + '\n')
            test_snt_space_f.write(snts_space[idx * 10 + gap] + '\n')
            snts_space.pop(idx * 10 + gap)
            snts_space.pop(idx * 10)
            dev_pos_f.write(poses[idx * 10] + '\n')
            test_pos_f.write(poses[idx * 10 + gap] + '\n')
            poses.pop(idx * 10 + gap)
            poses.pop(idx * 10)

        dev_amr_f.close()
        test_amr_f.close()
        dev_id_f.close()
        test_id_f.close()
        dev_snt_f.close()
        test_snt_f.close()
        dev_snt_space_f.close()
        test_snt_space_f.close()
        dev_pos_f.close()
        test_pos_f.close()

        left_num = len(amrs)
        print("%d %d %d" % (original_num, split, left_num))

        amr_f = open(amr_file, 'w', encoding='UTF-8')
        amr_f.write('\n'.join(amrs))
        amr_f.close()
        id_f = open(id_file, 'w', encoding='UTF-8')
        id_f.write('\n'.join(ids))
        id_f.close()
        snt_f = open(snt_file, 'w', encoding='UTF-8')
        snt_f.write('\n'.join(snts))
        snt_f.close()
        snt_space_f = open(snt_space_file, 'w', encoding='UTF-8')
        snt_space_f.write('\n'.join(snts_space))
        snt_space_f.close()
        pos_f = open(pos_file, 'w', encoding='UTF-8')
        pos_f.write('\n'.join(poses))
        pos_f.close()


get_all_data()
create_dev_test()
