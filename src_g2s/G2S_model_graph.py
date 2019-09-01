import tensorflow as tf
import graph_encoder_utils
import generator_utils
import padding_utils
from tensorflow.python.ops import variable_scope
import numpy as np
import random

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
cc = SmoothingFunction()

class ModelGraph(object):
    def __init__(self, word_vocab, char_vocab, Edgelabel_vocab, options=None, mode='ce_train'):
        # here 'mode', whose value can be:
        #  'ce_train',
        #  'rl_train',
        #  'evaluate',
        #  'evaluate_bleu',
        #  'decode'.
        # it is different from 'mode_gen' in generator_utils.py
        # value of 'mode_gen' can be ['ce_loss', 'rl_loss', 'greedy' or 'sample']
        self.mode = mode
        # if self.mode == 'decode':
        #     self.decode_flag = True
        # is_training controls whether to use dropout
        is_training = True if mode in ('ce_train', ) else False

        self.options = options
        self.word_vocab = word_vocab


        # encode the input instance
        # encoder.graph_hidden [batch, node_num, vsize]
        # encoder.graph_cell [batch, node_num, vsize]
        self.encoder = graph_encoder_utils.GraphEncoder(
                word_vocab = word_vocab,
                edge_label_vocab = Edgelabel_vocab,
                char_vocab = char_vocab,
                is_training = is_training, options = options)

        # ============== Choices of attention memory ================
        if options.attention_type == 'hidden':
            self.encoder_dim = options.neighbor_vector_dim
            self.encoder_states = self.encoder.graph_hiddens
        elif options.attention_type == 'hidden_cell':
            self.encoder_dim = options.neighbor_vector_dim * 2
            self.encoder_states = tf.concat([self.encoder.graph_hiddens, self.encoder.graph_cells], 2)
        elif options.attention_type == 'hidden_embed':
            self.encoder_dim = options.neighbor_vector_dim + self.encoder.input_dim
            self.encoder_states = tf.concat([self.encoder.graph_hiddens, self.encoder.node_representations], 2)
        else:
            assert False, '%s not supported yet' % options.attention_type

        # ============== Choices of initializing decoder state =============
        if options.way_init_decoder == 'zero':
            new_c = tf.zeros([self.encoder.batch_size, options.gen_hidden_size])
            new_h = tf.zeros([self.encoder.batch_size, options.gen_hidden_size])
        elif options.way_init_decoder == 'all':
            new_c = tf.reduce_sum(self.encoder.graph_cells, axis=1)
            new_h = tf.reduce_sum(self.encoder.graph_hiddens, axis=1)
        elif options.way_init_decoder == 'root':
            new_c = self.encoder.graph_cells[:,0,:]
            new_h = self.encoder.graph_hiddens[:,0,:]
        else:
            assert False, 'way to initial decoder (%s) not supported' % options.way_init_decoder
        self.init_decoder_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        # prepare AMR-side input for decoder
        self.nodes = self.encoder.passage_nodes
        self.nodes_num = self.encoder.passage_nodes_size
        if options.with_char:
            self.nodes_chars = self.encoder.passage_nodes_chars
            self.nodes_chars_num = self.encoder.passage_nodes_chars_size
        self.nodes_mask = self.encoder.passage_nodes_mask

        self.in_neigh_indices = self.encoder.passage_in_neighbor_indices
        self.in_neigh_edges = self.encoder.passage_in_neighbor_edges
        self.in_neigh_mask = self.encoder.passage_in_neighbor_mask

        self.out_neigh_indices = self.encoder.passage_out_neighbor_indices
        self.out_neigh_edges = self.encoder.passage_out_neighbor_edges
        self.out_neigh_mask = self.encoder.passage_out_neighbor_mask

        self.create_placeholders(options)

        self.loss_weights = tf.sequence_mask(self.answer_len, options.max_answer_len, dtype=tf.float32) # [batch_size, gen_steps]
        self.loss_pos_weights = tf.sequence_mask(self.answer_pos_len, options.max_answer_len, dtype=tf.float32)  # [batch_size, gen_steps]

        with variable_scope.variable_scope("generator"):
            # create generator
            self.generator = generator_utils.CovCopyAttenGen(self, options, word_vocab)
            # calculate encoder_features
            self.encoder_features = self.generator.calculate_encoder_features(self.encoder_states, self.encoder_dim)

            if mode == 'decode':  # first decoder
                self.context_t_1 = tf.placeholder(tf.float32, [None, self.encoder_dim], name='context_t_1') # [batch_size, encoder_dim]
                self.coverage_t_1 = tf.placeholder(tf.float32, [None, None], name='coverage_t_1') # [batch_size, encoder_dim]
                self.word_t = tf.placeholder(tf.int32, [None], name='word_t') # [batch_size]

                (self.state_t, self.context_t, self.coverage_t, self.attn_dist_t, self.p_gen_t, self.ouput_t,
                    self.topk_log_probs, self.topk_ids, self.greedy_prediction, self.multinomial_prediction) = self.generator.decode_mode(
                        word_vocab, options.beam_size, self.init_decoder_state, self.context_t_1, self.coverage_t_1, self.word_t,
                        self.encoder_states, self.encoder_features, self.nodes, self.nodes_mask)

                self.create_decoder2_variable(options)

                self.context2_t_1 = tf.placeholder(tf.float32, [None, options.gen_hidden_size + self.encoder_dim],
                                                   name='context2_t_1')  # [batch_size, decoder_dim + encoder_dim]
                self.coverage2_t_1 = tf.placeholder(tf.float32, [None, None],
                                                    name='coverage2_t_1')  # [batch_size, decoder_dim]
                self.word2_t = tf.placeholder(tf.int32, [None], name='word2_t')  # [batch_size]
                self.decode1_idxs = tf.placeholder(tf.int32, [None, None], name='decode1_idxs')
                self.decode1_mask = tf.placeholder(tf.float32, [None, None], name='decode1_mask')

                ### any problem ???
                if options.reuse_decoder_state:
                    self.init_decoder2_state = self.init_decoder_state
                else:
                    new_c = tf.reduce_sum(self.decoder_c, axis=1)
                    new_h = tf.reduce_sum(self.decoder_h, axis=1)
                    self.init_decoder2_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

                (self.state2_t, self.context2_t, self.coverage2_t, self.attn_dist2_t, self.p_gen2_t, self.ouput2_t,
                 self.topk_log_probs2, self.topk_ids2, self.greedy_prediction2,
                 self.multinomial_prediction2) = self.generator.decode_mode_two_pass(
                    word_vocab, options.beam_size, self.init_decoder2_state, self.context2_t_1, self.coverage2_t_1,
                    self.word2_t, self.encoder_states, self.encoder_features, self.nodes, self.nodes_mask,
                    self.decoder1_states, self.decoder1_features, self.decode1_idxs, self.decode1_mask)

                # not buiding training op for this mode
                return
            elif mode == 'evaluate_bleu':
                _, loss1, self.greedy_words, self.decoder_c, self.decoder_h = self.generator.train_mode(word_vocab, self.encoder_dim,
                      self.encoder_states, self.encoder_features, self.nodes, self.nodes_mask, self.init_decoder_state,self.answer_pos_inp,
                      self.answer_pos_ref, self.loss_weights, mode_gen='greedy')
                self.create_decoder2_variable(options)
                _, _, self.greedy_words,  = self.generator.train_mode_two_pass(word_vocab, self.encoder_dim,
                      self.encoder_states, self.encoder_features, self.decoder1_dim, self.decoder1_states, self.decoder1_features, self.nodes, self.nodes_mask,
                      self.decode1_idxs, self.decode1_mask, self.init_decoder2_state, self.answer_inp, self.answer_ref, self.loss_weights, loss1, None, mode_gen='greedy')
                # not buiding training op for this mode
                return
            elif mode in ('ce_train', 'evaluate', ):
                accu1, self.vocab_score1, loss1, _, _, _ = self.generator.train_mode(word_vocab, self.encoder_dim,
                    self.encoder_states, self.encoder_features, self.nodes, self.nodes_mask, self.init_decoder_state, self.answer_pos_inp,
                    self.answer_pos_ref, self.loss_weights, mode_gen='ce_loss')
                # tf.get_variable_scope().reuse_variables()
                _, _, self.greedy_words, self.decoder_c, self.decoder_h = self.generator.train_mode(word_vocab, self.encoder_dim,
                    self.encoder_states,self.encoder_features, self.nodes, self.nodes_mask, self.init_decoder_state,
                    self.answer_pos_inp, self.answer_pos_ref, self.loss_weights, mode_gen='greedy')
                if loss1 is None:
                    print('loss1 is None!')
                self.create_decoder2_variable(options)
                self.accu, self.vocab_score2, self.loss, _ = self.generator.train_mode_two_pass(word_vocab, self.encoder_dim,
                    self.encoder_states, self.encoder_features, self.decoder1_dim, self.decoder1_states, self.decoder1_features, self.nodes, self.nodes_mask,
                    self.decode1_idxs, self.decode1_mask, self.init_decoder2_state, self.answer_inp, self.answer_ref, self.loss_weights, loss1, accu1, mode_gen='ce_loss')

                if mode == 'evaluate': return # not buiding training op for evaluation
            elif mode == 'rl_train':  # how to use twopass in rl train ???
                _, _, _, self.decoder_c, self.decoder_h = self.generator.train_mode(word_vocab, self.encoder_dim,
                    self.encoder_states,self.encoder_features, self.nodes, self.nodes_mask, self.init_decoder_state,
                    self.answer_pos_inp, self.answer_pos_ref, self.loss_weights, mode_gen='greedy')
                _, loss11, _, _, _ = self.generator.train_mode(word_vocab, self.encoder_dim,
                    self.encoder_states,self.encoder_features, self.nodes, self.nodes_mask, self.init_decoder_state,
                    self.answer_pos_inp, self.answer_pos_ref, self.loss_weights, mode_gen='rl_loss')
                self.create_decoder2_variable(options)
                _, self.loss, _ = self.generator.train_mode_two_pass(word_vocab, self.encoder_dim, self.encoder_states, self.encoder_features,
                    self.decoder1_dim, self.decoder1_states, self.decoder1_features, self.nodes, self.nodes_mask, self.decode1_idxs, self.decode1_mask,
                    self.init_decoder2_state, self.answer_inp, self.answer_ref, self.loss_weights, loss11, None, mode_gen='rl_loss')

                tf.get_variable_scope().reuse_variables()

                _, loss12, self.sampled_words, self.decoder_c, self.decoder_h = self.generator.train_mode(word_vocab, self.encoder_dim,
                    self.encoder_states, self.encoder_features, self.nodes, self.nodes_mask, self.init_decoder_state,
                    self.answer_pos_inp, self.answer_pos_ref, None, mode_gen='sample')
                self.create_decoder2_variable(options)
                _, _, self.sampled_words = self.generator.train_mode_two_pass(word_vocab, self.encoder_dim,
                    self.encoder_states, self.encoder_features, self.decoder1_dim, self.decoder1_states,self.decoder1_features, self.nodes, self.nodes_mask,
                    self.decode1_nodes, self.decode1_mask, self.init_decoder2_state, self.answer_inp, self.answer_ref, None, loss12, None, mode_gen='sample')

                _, loss13, self.greedy_words, self.decoder_c, self.decoder_h = self.generator.train_mode(word_vocab, self.encoder_dim,
                    self.encoder_states,self.encoder_features, self.nodes, self.nodes_mask, self.init_decoder_state,
                    self.answer_pos_inp, self.answer_pos_ref, None, mode_gen='greedy')
                self.create_decoder2_variable(options)
                _, _, self.greedy_words = self.generator.train_mode_two_pass(word_vocab, self.encoder_dim,
                    self.encoder_states, self.encoder_features, self.decoder1_dim, self.decoder1_states,self.decoder1_features, self.nodes, self.nodes_mask,
                    self.decode1_nodes, self.decode1_mask, self.init_decoder2_state, self.answer_inp, self.answer_ref, None, loss13, None, mode_gen='greedy')

        if options.optimize_type == 'adadelta':
            clipper = 50
            if options.lr_decay:
                self.current_step = tf.Variable(tf.constant(0))
                self.decay_steps = tf.Variable(tf.constant(2880))
                self.decay_rate = 0.5
                self.lr = tf.train.exponential_decay(options.learning_rate, self.current_step, self.decay_steps, self.decay_rate)
            else:
                self.lr = tf.Variable(options.learning_rate)
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
            tvars = tf.trainable_variables()
            if options.lambda_l2>0.0:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
                self.loss = self.loss + options.lambda_l2 * l2_loss
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        elif options.optimize_type == 'adam':
            clipper = 50
            if options.lr_decay:
                self.current_step = tf.Variable(tf.constant(0))
                self.decay_steps = tf.Variable(tf.constant(2880))
                self.decay_rate = 0.5
                self.lr = tf.train.exponential_decay(options.learning_rate, self.current_step, self.decay_steps, self.decay_rate)
            else:
                self.lr = tf.Variable(options.learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            tvars = tf.trainable_variables()
            if options.lambda_l2>0.0:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
                self.loss = self.loss + options.lambda_l2 * l2_loss
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        extra_train_ops = []
        train_ops = [self.train_op] + extra_train_ops
        self.train_op = tf.group(*train_ops)

    def create_placeholders(self, options):
        # build placeholder for answer
        self.answer_ref = tf.placeholder(tf.int32, [None, options.max_answer_len], name="answer_ref") # [batch_size, gen_steps]
        self.answer_inp = tf.placeholder(tf.int32, [None, options.max_answer_len], name="answer_inp") # [batch_size, gen_steps]
        self.answer_pos_ref = tf.placeholder(tf.int32, [None, options.max_answer_len], name="answer_ref")  # [batch_size, gen_steps]
        self.answer_pos_inp = tf.placeholder(tf.int32, [None, options.max_answer_len], name="answer_inp")  # [batch_size, gen_steps]

        self.answer_len = tf.placeholder(tf.int32, [None], name="answer_len") # [batch_size]
        self.answer_pos_len = tf.placeholder(tf.int32, [None], name="answer_pos_len")  # [batch_size]

        self.decoder_c = tf.placeholder(tf.float32, [None,None,options.gen_hidden_size], name="decoder_c")
        self.decoder_h = tf.placeholder(tf.float32, [None,None,options.gen_hidden_size], name='decoder_h')

        # # prepare for second decode
        # self.nodes1 = tf.placeholder(tf.int32, [None, None])  # [batch_size, answer_nodes_size_max]
        # self.nodes_mask1 = tf.placeholder(tf.float32, [None, None])

        # build placeholder for reinforcement learning
        self.reward = tf.placeholder(tf.float32, [None], name="reward")

    def create_decoder2_variable(self, options):
        '''
        decoder_c :[batch_size, answer1_len, gen_hidden_dize]
        decoder_h :[batch_size, answer1_len, gen_hidden_dize]
        '''
        # ============== Choices of attention memory ================
        self.decoder1_dim = options.gen_hidden_size
        self.decoder1_states = self.decoder_h
        self.decoder1_features = self.generator.calculate_decoder_features(self.decoder1_states, self.decoder1_dim)
        # ============== Choices of initializing decoder state =============
        if options.mode is not 'decode':
            self.init_decoder2_state = None
            if options.reuse_decoder_state:
                self.init_decoder2_state = self.init_decoder_state
            else:
                new_c = tf.reduce_sum(self.decoder_c, axis=1)
                new_h = tf.reduce_sum(self.decoder_h, axis=1)
                self.init_decoder2_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            # ============== create new decode1_idxs and decode1_mask =============
            self.decode1_idxs = self.answer_pos_inp
            self.decode1_size = self.answer_pos_len
            seqs_size_max = self.decode1_idxs.shape[1]
            self.decode1_mask = tf.sequence_mask(self.decode1_size, seqs_size_max, dtype=tf.float32)

    def run_decoder2(self, sess, decoder1_cell_state, decoder1_hidden_state):
        feed_dict = {}
        feed_dict[self.decoder_c] = decoder1_cell_state
        feed_dict[self.decoder_h] = decoder1_hidden_state
        return sess.run([self.decoder1_states, self.decoder1_features], feed_dict)

    def run_greedy(self, sess, batch, options):
        feed_dict = self.run_encoder(sess, batch, options, only_feed_dict=True) # reuse this function to construct feed_dict
        feed_dict[self.answer_inp] = batch.sent_inp
        if options.pos_mask:
            feed_dict[self.answer_pos_inp] = batch.sent_pos_inp
        else:
            feed_dict[self.answer_pos_inp] = batch.sent_inp
        return sess.run(self.greedy_words, feed_dict)

    def run_ce_training(self, sess, batch, options, current_step=0, decay_steps=3000, only_eval=False):
        feed_dict = self.run_encoder(sess, batch, options, only_feed_dict=True) # reuse this function to construct feed_dict
        feed_dict[self.answer_inp] = batch.sent_inp
        feed_dict[self.answer_ref] = batch.sent_out
        if not only_eval and options.lr_decay:
            feed_dict[self.current_step] = current_step # tf.Variable(tf.constant(current_step))
            feed_dict[self.decay_steps] = decay_steps # tf.Variable(tf.constant(decay_steps))

        if options.pos_mask:
            feed_dict[self.answer_pos_inp] = batch.sent_pos_inp
            feed_dict[self.answer_pos_ref] = batch.sent_pos_out
            feed_dict[self.answer_pos_len] = batch.sent_pos_len
        else:
            feed_dict[self.answer_pos_inp] = batch.sent_inp
            feed_dict[self.answer_pos_ref] = batch.sent_out
            feed_dict[self.answer_pos_len] = batch.sent_len

        feed_dict[self.answer_len] = batch.sent_len

        if only_eval:
            # return sess.run([self.accu, self.loss], feed_dict)
            return sess.run([self.accu, self.loss, self.vocab_score1, self.vocab_score2, self.greedy_words, self.loss_weights], feed_dict)
        else:
            return sess.run([self.train_op, self.loss, self.lr], feed_dict)[1:]


    def run_rl_training_subsample(self, sess, batch, options, current_step=0, decay_steps=3000):  # how to edit the rl training???
        flipp = options.flipp if options.__dict__.has_key('flipp') else 0.1

        # make feed_dict
        feed_dict = self.run_encoder(sess, batch, options, only_feed_dict=True)
        feed_dict[self.answer_inp] = batch.sent_inp
        if options.pos_mask:
            feed_dict[self.answer_pos_inp] = batch.sent_pos_inp
        feed_dict[self.answer_pos_inp] = batch.sent_inp
        if options.lr_decay:
            feed_dict[self.current_step] = tf.Variable(tf.constant(current_step))
            feed_dict[self.decay_steps] = tf.Variable(tf.constant(decay_steps))

        # get greedy and gold outputs
        greedy_output = sess.run(self.greedy_words, feed_dict) # [batch, sent_len]
        greedy_output = greedy_output.tolist()
        gold_output = batch.sent_out.tolist()

        # generate sample_output by flipping coins
        sample_output = np.copy(batch.sent_out)
        for i in range(batch.sent_out.shape[0]):
            seq_len = min(options.max_answer_len, batch.sent_len[i]-1) # don't change stop token '</s>'
            for j in range(seq_len):
                if greedy_output[i][j] != 0 and random.random() < flipp:
                    sample_output[i,j] = greedy_output[i][j]
        sample_output = sample_output.tolist()

        st_wid = self.word_vocab.getIndex('<s>')
        en_wid = self.word_vocab.getIndex('</s>')

        rl_inputs = []
        rl_outputs = []
        rl_input_lengths = []
        reward = []
        for i, (sout,gout) in enumerate(zip(sample_output,greedy_output)):
            sout, slex = self.word_vocab.getLexical(sout)
            gout, glex = self.word_vocab.getLexical(gout)
            rl_inputs.append([st_wid,]+sout[:-1])
            rl_outputs.append(sout)
            rl_input_lengths.append(len(sout))
            _, ref_lex = self.word_vocab.getLexical(gold_output[i])
            slst = slex.split()
            glst = glex.split()
            rlst = ref_lex.split()
            if options.reward_type == 'bleu':
                r = sentence_bleu([rlst], slst, smoothing_function=cc.method3)
                b = sentence_bleu([rlst], glst, smoothing_function=cc.method3)
            elif options.reward_type == 'rouge':
                r = sentence_rouge(ref_lex, slex, smoothing_function=cc.method3)
                b = sentence_rouge(ref_lex, glex, smoothing_function=cc.method3)
            reward.append(r-b)
            #print('Ref: {}'.format(ref_lex.encode('utf-8','ignore')))
            #print('Sample: {}'.format(slex.encode('utf-8','ignore')))
            #print('Greedy: {}'.format(glex.encode('utf-8','ignore')))
            #print('R-B: {}'.format(reward[-1]))
            #print('-----')

        rl_inputs = padding_utils.pad_2d_vals(rl_inputs, len(rl_inputs), self.options.max_answer_len)
        rl_outputs = padding_utils.pad_2d_vals(rl_outputs, len(rl_outputs), self.options.max_answer_len)
        rl_input_lengths = np.array(rl_input_lengths, dtype=np.int32)
        reward = np.array(reward, dtype=np.float32)
        assert rl_inputs.shape == rl_outputs.shape

        feed_dict = self.run_encoder(sess, batch, options, only_feed_dict=True)
        feed_dict[self.reward] = reward
        feed_dict[self.answer_inp] = rl_inputs
        feed_dict[self.answer_ref] = rl_outputs
        feed_dict[self.answer_len] = rl_input_lengths
        # feed_dict[self.answer_pos_len] = ??

        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


    def run_rl_training_model(self, sess, batch, options):  # what's this???
        feed_dict = self.run_encoder(sess, batch, options, only_feed_dict=True)
        feed_dict[self.answer_inp] = batch.sent_inp
        if options.pos_mask:
            feed_dict[self.answer_pos_inp] = batch.sent_pos_inp
        else:
            feed_dict[self.answer_pos_inp] = batch.sent_inp

        sample_output, greedy_output = sess.run(
                [self.sampled_words, self.greedy_words], feed_dict)

        sample_output = sample_output.tolist()
        greedy_output = greedy_output.tolist()

        st_wid = self.word_vocab.getIndex('<s>')
        en_wid = self.word_vocab.getIndex('</s>')

        rl_inputs = []
        rl_outputs = []
        rl_input_lengths = []
        reward = []
        for i, (sout,gout) in enumerate(zip(sample_output,greedy_output)):
            sout, slex = self.word_vocab.getLexical(sout)
            gout, glex = self.word_vocab.getLexical(gout)
            rl_inputs.append([st_wid,]+sout[:-1])
            rl_outputs.append(sout)
            rl_input_lengths.append(len(sout))
            ref_lex = batch.instances[i][-1]
            #r = metric_utils.evaluate_captions([ref_lex,],[slex,])
            #b = metric_utils.evaluate_captions([ref_lex,],[glex,])
            slst = slex.split()
            glst = glex.split()
            rlst = ref_lex.split()
            if options.reward_type == 'bleu':
                r = sentence_bleu([rlst], slst, smoothing_function=cc.method3)
                b = sentence_bleu([rlst], glst, smoothing_function=cc.method3)
            elif options.reward_type == 'rouge':
                r = sentence_rouge(ref_lex, slex, smoothing_function=cc.method3)
                b = sentence_rouge(ref_lex, glex, smoothing_function=cc.method3)
            reward.append(r-b)
            #print('Ref: {}'.format(ref_lex.encode('utf-8','ignore')))
            #print('Sample: {}'.format(slex.encode('utf-8','ignore')))
            #print('Greedy: {}'.format(glex.encode('utf-8','ignore')))
            #print('R-B: {}'.format(reward[-1]))
            #print('-----')

        rl_inputs = padding_utils.pad_2d_vals(rl_inputs, len(rl_inputs), self.options.max_answer_len)
        rl_outputs = padding_utils.pad_2d_vals(rl_outputs, len(rl_outputs), self.options.max_answer_len)
        rl_input_lengths = np.array(rl_input_lengths, dtype=np.int32)
        reward = np.array(reward, dtype=np.float32)
        assert rl_inputs.shape == rl_outputs.shape

        feed_dict = self.run_encoder(sess, batch, options, only_feed_dict=True)
        feed_dict[self.reward] = reward
        feed_dict[self.answer_inp] = rl_inputs
        feed_dict[self.answer_out] = rl_outputs
        feed_dict[self.answer_len] = rl_input_lengths
        # feed_dict[self.answer_pos_len] = ??

        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

    def run_encoder(self, sess, batch, options, only_feed_dict=False):
        feed_dict = {}
        feed_dict[self.nodes] = batch.nodes
        feed_dict[self.nodes_num] = batch.node_num
        if options.with_char:
            feed_dict[self.nodes_chars] = batch.nodes_chars
            feed_dict[self.nodes_chars_num] = batch.nodes_chars_num

        feed_dict[self.in_neigh_indices] = batch.in_neigh_indices
        feed_dict[self.in_neigh_edges] = batch.in_neigh_edges
        feed_dict[self.in_neigh_mask] = batch.in_neigh_mask

        feed_dict[self.out_neigh_indices] = batch.out_neigh_indices
        feed_dict[self.out_neigh_edges] = batch.out_neigh_edges
        feed_dict[self.out_neigh_mask] = batch.out_neigh_mask

        if only_feed_dict:
            return feed_dict

        return sess.run([self.encoder_states, self.encoder_features, self.nodes, self.nodes_mask, self.init_decoder_state],
                feed_dict)

if __name__ == '__main__':
    summary = " Tokyo is the one of the biggest city in the world."
    reference = "The capital of Japan, Tokyo, is the center of Japanese economy."

