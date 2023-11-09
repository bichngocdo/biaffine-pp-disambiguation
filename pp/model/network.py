import os

import core.nn
import numpy as np
import tensorflow as tf

import pp.bert.model as bert_modeling
from bert.modeling import BertConfig, BertModel
from pp.utils.ranking_loss import ranking_loss_all, ranking_loss_max


class PPDisambiguator(object):
    def __init__(self, args):
        self.ignore_variables = list()

        self._build_embeddings(args)

        self.train, self.forward, self.backward = self._build_train_functions(args)
        self.eval = self._build_eval_function(args)

        self.make_train_summary = self._build_train_summary_function()
        self.make_dev_summary = self._build_dev_summary_function()

    def initialize_global_variables(self, session):
        feed_dict = dict()
        if self.word_pt_embeddings is not None:
            feed_dict[self.word_pt_embeddings_ph] = self._word_pt_embeddings
        if self.tag_pt_embeddings is not None:
            feed_dict[self.tag_pt_embeddings_ph] = self._tag_pt_embeddings
        session.run(tf.global_variables_initializer(), feed_dict=feed_dict)

    def _build_train_summary_function(self):
        with tf.variable_scope('train_summary/'):
            x_acc = tf.placeholder(tf.float32,
                                   shape=None,
                                   name='x_acc')
            x_sent_acc = tf.placeholder(tf.float32,
                                        shape=None,
                                        name='x_sent_acc')

            tf.summary.scalar('train_acc', x_acc, collections=['train_summary'])
            tf.summary.scalar('train_sent_acc', x_sent_acc, collections=['train_summary'])

            summary = tf.summary.merge_all(key='train_summary')

        def f(session, acc, sent_acc):
            feed_dict = {
                x_acc: acc,
                x_sent_acc: sent_acc,
            }
            return session.run(summary, feed_dict=feed_dict)

        return f

    def _build_dev_summary_function(self):
        with tf.variable_scope('dev_summary/'):
            x_loss = tf.placeholder(tf.float32,
                                    shape=None,
                                    name='x_loss')
            x_acc = tf.placeholder(tf.float32,
                                   shape=None,
                                   name='x_clf_acc')
            x_sent_acc = tf.placeholder(tf.float32,
                                        shape=None,
                                        name='x_att_acc')

            tf.summary.scalar('dev_loss', x_loss, collections=['dev_summary'])
            tf.summary.scalar('dev_acc', x_acc, collections=['dev_summary'])
            tf.summary.scalar('dev_sent_acc', x_sent_acc, collections=['dev_summary'])

            summary = tf.summary.merge_all(key='dev_summary')

        def f(session, loss, acc, sent_acc):
            feed_dict = {
                x_loss: loss,
                x_acc: acc,
                x_sent_acc: sent_acc,
            }
            return session.run(summary, feed_dict=feed_dict)

        return f

    def _build_placeholders(self, args):
        # x_word has shape (batch_size, max_length)
        x_word = tf.placeholder(tf.int32,
                                shape=(None, None),
                                name='x_word')
        if self.word_pt_embeddings is not None:
            x_pt_word = tf.placeholder(tf.int32,
                                       shape=(None, None),
                                       name='x_pt_word')
        else:
            x_pt_word = None

        # x_tag has shape (batch_size, max_length)
        x_tag = tf.placeholder(tf.int32,
                               shape=(None, None),
                               name='x_tag')
        if self.tag_pt_embeddings is not None:
            x_pt_tag = tf.placeholder(tf.int32,
                                      shape=(None, None),
                                      name='x_pt_tag')
        else:
            x_pt_tag = None

        # x_char
        x_char = None

        # x_topo has shape (batch_size, max_length)
        if args.use_topological_fields:
            x_topo = tf.placeholder(tf.int32,
                                    shape=(None, None),
                                    name='x_topo')
        else:
            x_topo = None

        # x_bert_ids has shape (batch_size, max_token_length)
        # x_bert_mask has shape (batch_size, max_token_length)
        # x_bert_types has shape (batch_size, max_token_length)
        # x_bert_indices has shape (batch_size, max_length, max_subword_length)
        if args.bert_path is not None:
            x_bert_id = tf.placeholder(tf.int32,
                                       shape=(None, None),
                                       name='x_bert_ids')
            x_bert_mask = tf.placeholder(tf.int32,
                                         shape=(None, None),
                                         name='x_bert_mask')
            x_bert_type = tf.placeholder(tf.int32,
                                         shape=(None, None),
                                         name='x_bert_types')
            x_bert_index = tf.placeholder(tf.int32,
                                          shape=(None, None, None),
                                          name='x_bert_indices')
        else:
            x_bert_id = None
            x_bert_mask = None
            x_bert_type = None
            x_bert_index = None

        # x_prep has shape (batch_size, num_tuples)
        x_prep = tf.placeholder(tf.int32,
                                shape=(None, None),
                                name='x_prep')

        # x_obj has shape (batch_size, num_tuples)
        x_obj = tf.placeholder(tf.int32,
                               shape=(None, None),
                               name='x_obj')

        # x_scores has shape (batch_size, num_tuples, max_length, 5)
        if args.use_scores:
            x_scores = tf.placeholder(tf.float32,
                                      shape=(None, None, None, 5),
                                      name='x_scores')
        else:
            x_scores = None

        # y_head has shape (batch_size, num_tuples)
        y_head = tf.placeholder(tf.int32,
                                shape=(None, None),
                                name='y_head')
        return [x_word, x_pt_word, x_tag, x_pt_tag, x_char, x_topo,
                x_bert_id, x_bert_mask, x_bert_type, x_bert_index,
                x_prep, x_obj, x_scores, y_head]

    def _build_embeddings(self, args):
        with tf.variable_scope('embeddings'):
            if not args.bert_path:
                if args.word_embeddings is not None:
                    self.word_embeddings = tf.get_variable('word_embeddings',
                                                           shape=(args.no_words, args.word_dim),
                                                           dtype=tf.float32,
                                                           initializer=tf.zeros_initializer,
                                                           regularizer=tf.contrib.layers.l2_regularizer(args.word_l2))
                    self.word_pt_embeddings_ph = tf.placeholder(tf.float32,
                                                                shape=args.word_embeddings.shape,
                                                                name='word_pt_embeddings_ph')
                    self.word_pt_embeddings = tf.Variable(self.word_pt_embeddings_ph,
                                                          name='word_pt_embeddings',
                                                          trainable=False)
                    self._word_pt_embeddings = args.word_embeddings
                else:
                    self.word_embeddings = \
                        tf.get_variable('word_embeddings',
                                        shape=(args.no_words, args.word_dim),
                                        dtype=tf.float32,
                                        initializer=tf.random_normal_initializer,
                                        regularizer=tf.contrib.layers.l2_regularizer(args.word_l2))
                    self.word_pt_embeddings = None
            else:
                if args.word_dim > 0:
                    self.word_embeddings = tf.get_variable('word_embeddings',
                                                           shape=(args.no_words, args.word_dim),
                                                           dtype=tf.float32,
                                                           initializer=tf.random_normal_initializer,
                                                           regularizer=tf.contrib.layers.l2_regularizer(args.word_l2))
                else:
                    self.word_embeddings = None
                self.word_pt_embeddings = None

            if args.tag_embeddings is not None:
                self.tag_embeddings = tf.get_variable('tag_embeddings',
                                                      shape=(args.no_tags, args.tag_dim),
                                                      dtype=tf.float32,
                                                      initializer=tf.zeros_initializer,
                                                      regularizer=tf.contrib.layers.l2_regularizer(args.tag_l2))
                self.tag_pt_embeddings_ph = tf.placeholder(tf.float32,
                                                           shape=args.tag_embeddings.shape,
                                                           name='tag_pt_embeddings_ph')
                self.tag_pt_embeddings = tf.Variable(self.tag_pt_embeddings_ph,
                                                     name='tag_pt_embeddings',
                                                     trainable=False)
                self._tag_pt_embeddings = args.tag_embeddings
            else:
                self.tag_embeddings = \
                    tf.get_variable('tag_embeddings',
                                    shape=(args.no_tags, args.tag_dim),
                                    dtype=tf.float32,
                                    initializer=tf.random_normal_initializer,
                                    regularizer=tf.contrib.layers.l2_regularizer(args.tag_l2))
                self.tag_pt_embeddings = None

            if args.use_topological_fields:
                self.topological_field_embeddings = \
                    tf.get_variable('topological_field_embeddings',
                                    shape=(args.no_topological_fields, args.topological_field_dim),
                                    dtype=tf.float32,
                                    initializer=tf.random_normal_initializer,
                                    regularizer=tf.contrib.layers.l2_regularizer(args.l2))
            else:
                self.topological_field_embeddings = None

    def _build_input_layers(self, args, is_training):
        def f(x_word, x_pt_word, x_tag, x_pt_tag, x_char, x_topo,
              x_bert_id, x_bert_mask, x_bert_type, x_bert_index):
            if not args.bert_path:
                # Word embeddings
                e_word = tf.nn.embedding_lookup(self.word_embeddings, x_word)

                # Word pre-trained embeddings
                if self.word_pt_embeddings is not None:
                    e_pt_word = tf.nn.embedding_lookup(self.word_pt_embeddings, x_pt_word)
                    e_word += e_pt_word

                if is_training:
                    e_word = tf.nn.dropout(e_word,
                                           keep_prob=1 - args.input_dropout,
                                           noise_shape=core.nn.noise_shape(e_word, (None, None, 1)))
            else:
                # Word embeddings
                if self.word_embeddings is not None:
                    e_word = tf.nn.embedding_lookup(self.word_embeddings, x_word)
                else:
                    e_word = None

                # BERT embeddings
                bert_layer = self._build_bert_layer(args, is_training,
                                                    trainable=args.bert_fine_tuning)
                e_subword = bert_layer(x_bert_id, x_bert_mask, x_bert_type)

                batch_size, max_length, max_word_length = tf.unstack(tf.shape(x_bert_index))
                dim = e_subword.get_shape()[-1]

                index_mask = tf.greater_equal(x_bert_index, 0)
                indices = tf.where(index_mask, x_bert_index, tf.zeros_like(x_bert_index))

                indices = tf.reshape(indices, (batch_size, max_length * max_word_length))
                e_subword = tf.batch_gather(e_subword, indices)
                e_subword = tf.reshape(e_subword, (batch_size, max_length, max_word_length, dim))
                subword_mask = tf.tile(tf.expand_dims(index_mask, -1), (1, 1, 1, dim))
                e_subword = tf.where(subword_mask, e_subword, tf.zeros_like(e_subword))
                no_subwords = tf.reduce_sum(tf.cast(subword_mask, tf.float32), axis=-2)
                e_word_bert = tf.reduce_sum(e_subword, axis=-2) / no_subwords
                e_word_bert = tf.where(tf.not_equal(no_subwords, 0), e_word_bert, tf.zeros_like(e_word_bert))

                if self.word_embeddings is not None:
                    e_word = tf.concat([e_word, e_word_bert], axis=-1)
                else:
                    e_word = e_word_bert

                if is_training:
                    e_word = tf.nn.dropout(e_word,
                                           keep_prob=1 - args.input_dropout,
                                           noise_shape=core.nn.noise_shape(e_word, (None, None, 1)))

            # Tag embeddings
            e_tag = tf.nn.embedding_lookup(self.tag_embeddings, x_tag)

            # Tag pre-trained embeddings
            if self.tag_pt_embeddings is not None:
                e_pt_tag = tf.nn.embedding_lookup(self.tag_pt_embeddings, x_pt_tag)
                e_tag += e_pt_tag

            if is_training:
                e_tag = tf.nn.dropout(e_tag,
                                      keep_prob=1 - args.input_dropout,
                                      noise_shape=core.nn.noise_shape(e_tag, (None, None, 1)))

            if args.use_topological_fields:
                e_topo = tf.nn.embedding_lookup(self.topological_field_embeddings, x_topo)
                if is_training:
                    e_topo = tf.nn.dropout(e_topo,
                                           keep_prob=1 - args.input_dropout,
                                           noise_shape=core.nn.noise_shape(e_topo, (None, None, 1)))
                input = tf.concat([e_word, e_tag, e_topo], axis=-1)
            else:
                input = tf.concat([e_word, e_tag], axis=-1)

            return input

        return f

    def _build_bert_layer(self, args, is_training, trainable=True):
        def f(x_bert_id, x_bert_mask, x_bert_type):
            bert_config_file = os.path.join(args.bert_path, 'bert_config.json')
            bert_checkpoint = os.path.join(args.bert_path, 'bert_model.ckpt')
            bert_config = BertConfig.from_json_file(bert_config_file)
            with tf.variable_scope('bert'):
                model = BertModel(bert_config, is_training=is_training,
                                  input_ids=x_bert_id, input_mask=x_bert_mask, token_type_ids=x_bert_type)
                bert_modeling.initialize_from_checkpoint(bert_checkpoint)
                if not trainable:
                    self.ignore_variables += tf.trainable_variables(tf.get_variable_scope().name)
            all_layers = model.get_all_encoder_layers()
            selected_layers = [all_layers[i] for i in args.bert_layers]
            return tf.add_n(selected_layers) / len(args.bert_layers)

        return f

    def _build_hidden_layers(self, args, is_training):
        def f(input, lengths):
            hidden = input
            with tf.variable_scope('lstms'):
                for i in range(args.num_lstms):
                    if is_training:
                        hidden = tf.nn.dropout(hidden,
                                               keep_prob=1 - args.dropout,
                                               noise_shape=core.nn.noise_shape(hidden, (None, 1, None)))
                    fw_cell = tf.nn.rnn_cell.LSTMCell(args.hidden_dim,
                                                      initializer=tf.initializers.orthogonal)
                    if is_training:
                        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell,
                                                                state_keep_prob=1 - args.recurrent_dropout,
                                                                variational_recurrent=True,
                                                                dtype=tf.float32)

                    bw_cell = tf.nn.rnn_cell.LSTMCell(args.hidden_dim,
                                                      initializer=tf.initializers.orthogonal)
                    if is_training:
                        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell,
                                                                state_keep_prob=1 - args.recurrent_dropout,
                                                                variational_recurrent=True,
                                                                dtype=tf.float32)
                    with tf.variable_scope('lstm%d' % i):
                        (fw, bw), (fw_s, bw_s) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, hidden,
                                                                                 sequence_length=lengths,
                                                                                 dtype=tf.float32)
                    hidden = tf.concat([fw, bw], axis=-1)

            with tf.variable_scope('mlp'):
                if is_training:
                    hidden = tf.nn.dropout(hidden,
                                           keep_prob=1 - args.dropout,
                                           noise_shape=core.nn.noise_shape(hidden, (None, 1, None)))
                for _ in range(args.num_mlps):
                    hidden = tf.layers.dense(hidden,
                                             units=3 * args.mlp_dim,
                                             activation=tf.nn.leaky_relu,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(args.l2))
                    if is_training:
                        hidden = tf.nn.dropout(hidden,
                                               keep_prob=1 - args.dropout,
                                               noise_shape=core.nn.noise_shape(hidden, (None, 1, None)))

            h_prep, h_obj, h_head = tf.split(hidden, 3, axis=-1)

            return h_prep, h_obj, h_head

        return f

    def _build_scoring_module(self, args, is_training):
        def f(hidden, x_prep, x_obj, x_scores):
            h_prep, h_obj, h_head = hidden
            if args.use_scores:
                x_scores = tf.pad(x_scores, [[0, 0], [0, 0], [1, 0], [0, 0]],
                                  mode='constant', constant_values=0.)
            if args.clf == 'bilinear':
                weight = tf.get_variable('weight',
                                         shape=(2 * args.mlp_dim + 1, args.mlp_dim),
                                         dtype=tf.float32,
                                         initializer=tf.zeros_initializer,
                                         regularizer=tf.contrib.layers.l2_regularizer(args.l2))

                prep_emb = tf.batch_gather(h_prep, x_prep, name='gather')  # shape (batch_size, num_tuples, hidden_dim)
                obj_mask = tf.greater(x_obj, 0)
                x_obj = tf.where(obj_mask, x_obj, tf.zeros_like(x_obj))
                obj_emb = tf.batch_gather(h_obj, x_obj, name='gather')  # shape (batch_size, num_tuples, hidden_dim)
                obj_emb *= tf.to_float(tf.expand_dims(obj_mask, axis=-1))
                prep_obj_emb = tf.concat([prep_emb, obj_emb], axis=-1)  # shape (batch_size, num_tuples, 2 * hidden_dim)

                prep_obj_emb = tf.pad(prep_obj_emb, [[0, 0], [0, 0], [0, 1]],
                                      mode='constant', constant_values=1.)
                # shape (batch_size, num_tuples, 2 * hidden_dim + 1)
                # biaffine h1 W h2 + w h2
                can_emb = h_head

                scores = core.nn.bilinear(prep_obj_emb, can_emb, weight)  # shape (batch_size, num_tuples, max_length)

                if args.use_scores:
                    x_scores = tf.layers.dense(x_scores,
                                               units=1,
                                               activation=tf.nn.leaky_relu,
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(args.l2))
                    scores += tf.squeeze(x_scores, axis=-1)
            elif args.clf == 'mlp':
                prep_emb = tf.batch_gather(h_prep, x_prep, name='gather')  # shape (batch_size, num_tuples, hidden_dim)
                obj_emb = tf.batch_gather(h_obj, x_obj, name='gather')  # shape (batch_size, num_tuples, hidden_dim)
                obj_mask = tf.greater(x_obj, 0)
                obj_emb *= tf.to_float(tf.expand_dims(obj_mask, axis=-1))

                prep_obj_emb = tf.concat([prep_emb, obj_emb], axis=-1)  # shape (batch_size, num_tuples, 2 * hidden_dim)
                prep_obj_emb = tf.expand_dims(prep_obj_emb, 2)  # shape (batch_size, num_tuples, 1, 2 * hidden_dim)
                prep_obj_emb = tf.tile(prep_obj_emb,
                                       tf.stack([1, 1, tf.shape(h_head)[1], 1]))
                # shape (batch_size, num_tuples, max_length, 2 * hidden_dim)

                can_emb = tf.expand_dims(h_head, 1)  # shape (batch_size, 1, max_length, hidden_dim)
                can_emb = tf.tile(can_emb, tf.stack([1, tf.shape(prep_obj_emb)[1], 1, 1]))
                # shape (batch_size, num_tuples, max_length, hidden_dim)

                emb = tf.concat([prep_obj_emb, can_emb], axis=-1)
                # shape (batch_size, num_tuples, max_length, 3 * hidden_dim)

                if args.use_scores:
                    emb = tf.concat([emb, x_scores], axis=-1)
                    # shape (batch_size, num_tuples, max_length, 3 * hidden_dim)

                emb = tf.layers.dense(emb,
                                      units=args.clf_hidden_dim,
                                      activation=tf.nn.leaky_relu,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(args.l2))
                scores = tf.layers.dense(emb,
                                         units=1,
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(args.l2))
                scores = tf.squeeze(scores, axis=-1)
                # shape (batch_size, num_tuples, max_length)
            else:
                raise Exception('Unknown classifier:', args.clf)

            return scores

        return f

    def _build(self, args, is_training):
        with tf.variable_scope('placeholders'):
            x_word, x_pt_word, x_tag, x_pt_tag, x_char, x_topo, \
                x_bert_id, x_bert_mask, x_bert_type, x_bert_index, \
                x_prep, x_obj, x_scores, y_head = self._build_placeholders(args)

        input_layers = self._build_input_layers(args, is_training)
        hidden_layers = self._build_hidden_layers(args, is_training)
        scoring_module = self._build_scoring_module(args, is_training)

        with tf.variable_scope('input_layers', reuse=tf.AUTO_REUSE):
            input = input_layers(x_word, x_pt_word, x_tag, x_pt_tag, x_char, x_topo,
                                 x_bert_id, x_bert_mask, x_bert_type, x_bert_index)
            mask = tf.greater(x_word, 0)
            lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)

        with tf.variable_scope('hidden_layers', reuse=tf.AUTO_REUSE):
            hidden = hidden_layers(input, lengths)  # shape (batch_size, max_length, hidden_dim)

        with tf.variable_scope('output_layers', reuse=tf.AUTO_REUSE):
            prep_mask = tf.greater(x_prep, 0)
            candidate_mask = tf.tile(tf.expand_dims(mask, 1),
                                     tf.stack([1, tf.shape(x_prep)[1], 1]))
            # shape (batch_size, num_tuples, max_length)
            logit = scoring_module(hidden, x_prep, x_obj, x_scores)  # shape (batch_size, num_tuples, max_length)
            logit = tf.where(candidate_mask,
                             logit,
                             tf.ones_like(logit) * np.float('-inf'))
            probability = tf.nn.softmax(logit)  # shape (batch_size, num_tuples, max_length)
            prediction = tf.argmax(probability, axis=-1, output_type=tf.int32)  # shape (batch_size, num_tuples)
            prediction = tf.where(prep_mask, prediction, tf.ones_like(prediction) * -1)

        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            if args.loss == 'cross_entropy':
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_head, logits=logit)
                no_instances = tf.reduce_sum(tf.to_float(prep_mask))
                loss = tf.reduce_mean(tf.boolean_mask(cross_entropy, prep_mask))
                loss = tf.cond(tf.equal(no_instances, 0), lambda: 0., lambda: loss)
            elif args.loss == 'ranking_all' or args.loss == 'ranking_max':
                max_length = tf.shape(x_word)[1]
                num_tuples = tf.shape(x_prep)[1]
                labels = tf.one_hot(y_head, max_length)  # shape (batch_size, num_tuples, max_length)
                label_mask = tf.logical_and(tf.tile(tf.expand_dims(mask, 1),
                                                    tf.stack([1, num_tuples, 1])),
                                            tf.tile(tf.expand_dims(prep_mask, -1),
                                                    tf.stack([1, 1, max_length])))
                labels = tf.where(label_mask, labels, tf.ones_like(labels) * -1)
                if args.loss == 'ranking_all':
                    loss = ranking_loss_all(probability, labels)
                elif args.loss == 'ranking_max':
                    loss = ranking_loss_max(probability, labels)
            else:
                raise Exception('Unknown loss:', args.loss)

        inputs = [x_word, x_pt_word, x_tag, x_pt_tag, x_char, x_topo,
                  x_bert_id, x_bert_mask, x_bert_type, x_bert_index,
                  x_prep, x_obj, x_scores, y_head]
        outputs = {
            'logit': logit,
            'probability': probability,
            'prediction': prediction
        }

        return inputs, outputs, loss

    def _build_train_function(self, args):
        with tf.name_scope('train'):
            inputs, outputs, loss = self._build(args, is_training=True)

            self.iteration = tf.Variable(-1, name='iteration', trainable=False)
            with tf.variable_scope('learning_rate'):
                self.learning_rate = tf.train.exponential_decay(learning_rate=args.learning_rate,
                                                                global_step=self.iteration,
                                                                decay_steps=args.decay_step,
                                                                decay_rate=args.decay_rate)

            with tf.variable_scope('train_summary/'):
                tf.summary.scalar('lr', self.learning_rate, collections=['train_instant_summary'])
                tf.summary.scalar('train_loss', loss, collections=['train_instant_summary'])
                summary = tf.summary.merge_all(key='train_instant_summary')

            if args.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif args.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            elif args.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif args.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            elif args.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            else:
                raise Exception('Unknown optimizer:', args.optimizer)

            gradients_vars = optimizer.compute_gradients(loss, tf.trainable_variables())
            if args.max_norm is not None:
                with tf.variable_scope('max_norm'):
                    gradients = [gv[0] for gv in gradients_vars]
                    gradients, _ = tf.clip_by_global_norm(gradients, args.max_norm)
                    gradients_vars = [(g, gv[1]) for g, gv in zip(gradients, gradients_vars)]

            with tf.variable_scope('optimizer'):
                train_step = optimizer.apply_gradients(gradients_vars, global_step=self.iteration)

        def f(vars, session):
            feed_dict = dict()
            for input, var in zip(inputs, vars):
                if input is not None:
                    feed_dict[input] = var
            _, output_, loss_, summary_ = session.run([train_step, outputs, loss, summary],
                                                      feed_dict=feed_dict)
            return output_, loss_, summary_

        return f

    def _build_train_functions(self, args):
        with tf.name_scope('train'):
            inputs, outputs, loss = self._build(args, is_training=True)
            trainable_variables = [v for v in tf.trainable_variables()
                                   if v not in self.ignore_variables]

            with tf.variable_scope('accumulation'):
                acc_loss = tf.Variable(0., name='acc_loss', trainable=False)
                acc_gradients = [tf.Variable(tf.zeros_like(var), trainable=False)
                                 for var in trainable_variables]
                acc_counter = tf.Variable(0., name='acc_counter', trainable=False)

            self.iteration = tf.Variable(-1, name='iteration', trainable=False)
            with tf.variable_scope('learning_rate'):
                self.learning_rate = tf.train.exponential_decay(learning_rate=args.learning_rate,
                                                                global_step=self.iteration,
                                                                decay_steps=args.decay_step,
                                                                decay_rate=args.decay_rate)

            with tf.variable_scope('train_summary/'):
                tf.summary.scalar('lr', self.learning_rate, collections=['train_instant_summary'])
                tf.summary.scalar('train_loss', acc_loss, collections=['train_instant_summary'])
                summary = tf.summary.merge_all(key='train_instant_summary')

            if args.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif args.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            elif args.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif args.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            elif args.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            else:
                raise Exception('Unknown optimizer:', args.optimizer)

            gradients_vars = optimizer.compute_gradients(loss, trainable_variables)
            for i in range(len(gradients_vars)):
                g, v = gradients_vars[i]
                if g is None:
                    gradients_vars[i] = (tf.zeros_like(v), v)
            gradients = [gv[0] for gv in gradients_vars]
            with tf.variable_scope('accumulation'):
                acc_gradients_ops = [tf.assign_add(acc_g, g)
                                     for acc_g, g in zip(acc_gradients, gradients)]
                acc_loss_ops = tf.assign_add(acc_loss, loss)
                acc_counter_ops = tf.assign_add(acc_counter, 1.)
                zero_gradients_ops = [tf.assign(acc_g, tf.zeros_like(v))
                                      for acc_g, v in zip(acc_gradients, trainable_variables)]
                zero_loss_ops = tf.assign(acc_loss, 0.)
                zero_counter_ops = tf.assign(acc_counter, 0.)

            acc_gradients_vars = [(g, v) for g, v in zip(acc_gradients, trainable_variables)]
            if args.max_norm is not None:
                with tf.variable_scope('max_norm'):
                    clipped_acc_gradients, _ = tf.clip_by_global_norm(acc_gradients, args.max_norm)
                    acc_gradients_vars = [(g, v) for g, v in zip(clipped_acc_gradients, trainable_variables)]
                    clipped_gradients, _ = tf.clip_by_global_norm(gradients, args.max_norm)
                    gradients_vars = [(g, gv[1]) for g, gv in zip(clipped_gradients, gradients_vars)]

            with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
                train_step = optimizer.apply_gradients(gradients_vars, global_step=self.iteration)
                forward_ops = [acc_loss_ops, acc_gradients_ops, acc_counter_ops]
                backward_ops = [tf.assign(acc_loss, acc_loss / acc_counter),
                                [tf.assign(g, (g / acc_counter)) for g in acc_gradients],
                                optimizer.apply_gradients(acc_gradients_vars, global_step=self.iteration)]
                reset_ops = [zero_loss_ops, zero_gradients_ops, zero_counter_ops]

        def f_train(vars, session):
            feed_dict = dict()
            for input, var in zip(inputs, vars):
                if input is not None:
                    feed_dict[input] = var
            _, output_, loss_, summary_ = session.run([train_step, outputs, loss, summary],
                                                      feed_dict=feed_dict)
            return output_, loss_, summary_

        def f_forward(vars, session):
            feed_dict = dict()
            for input, var in zip(inputs, vars):
                if input is not None:
                    feed_dict[input] = var
            _, output_, loss_, = session.run([forward_ops, outputs, loss],
                                             feed_dict=feed_dict)
            return output_, loss_

        def f_backward(session):
            _, loss_, summary_ = session.run([backward_ops, acc_loss, summary])
            session.run(reset_ops)
            return loss_, summary_

        return f_train, f_forward, f_backward

    def _build_eval_function(self, args):
        with tf.name_scope('eval'):
            inputs, outputs, loss = self._build(args, is_training=False)

        def f(vars, session):
            feed_dict = dict()
            for input, var in zip(inputs, vars):
                if input is not None:
                    feed_dict[input] = var
            return session.run([outputs, loss], feed_dict=feed_dict)

        return f
