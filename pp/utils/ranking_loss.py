import numpy as np
import tensorflow as tf


def compute_valid_mask(labels):
    negative_mask = tf.expand_dims(tf.equal(labels, 0), 2)
    positive_mask = tf.expand_dims(tf.equal(labels, 1), 1)
    valid_mask = tf.logical_and(negative_mask, positive_mask)
    return valid_mask


def ranking_loss_all(scores, labels, margin=1.):
    negative_scores = tf.expand_dims(scores, 2)
    positive_scores = tf.expand_dims(scores, 1)
    all_dists = negative_scores - positive_scores + margin

    valid_mask = compute_valid_mask(labels)
    valid_dists = tf.boolean_mask(all_dists, valid_mask)

    not_easy_mask = tf.greater(valid_dists, 0.)
    not_easy_dists = tf.boolean_mask(valid_dists, not_easy_mask)

    no_pairs = tf.reduce_sum(tf.to_float(not_easy_mask))
    loss = tf.reduce_sum(not_easy_dists) / no_pairs
    loss = tf.cond(tf.equal(no_pairs, 0), lambda: 0., lambda: loss)

    return loss


def ranking_loss_max(scores, labels, margin=1.):
    positive_mask = tf.equal(labels, 1)
    negative_mask = tf.equal(labels, 0)

    positive_scores = tf.where(positive_mask,
                               scores,
                               tf.ones_like(scores) * np.inf)
    positive_scores = tf.reduce_min(positive_scores, -1)

    negative_scores = tf.where(negative_mask,
                               scores,
                               tf.ones_like(scores) * -np.inf)
    negative_scores = tf.reduce_max(negative_scores, -1)

    losses = negative_scores - positive_scores + margin

    not_easy_mask = tf.greater(losses, 0.)
    not_easy_dists = tf.boolean_mask(losses, not_easy_mask)

    no_pairs = tf.reduce_sum(tf.to_float(not_easy_mask))
    loss = tf.reduce_sum(not_easy_dists) / no_pairs
    loss = tf.cond(tf.equal(no_pairs, 0), lambda: 0., lambda: loss)

    return loss


def soft_ranking_loss_max(scores, labels, positive_margin=2.5, negative_margin=0.5, gamma=2.):
    positive_mask = tf.equal(labels, 1)
    negative_mask = tf.equal(labels, 0)

    positive_scores = tf.where(positive_mask,
                               scores,
                               tf.ones_like(scores) * np.inf)
    positive_scores = tf.reduce_min(positive_scores, -1)

    negative_scores = tf.where(negative_mask,
                               scores,
                               tf.ones_like(scores) * -np.inf)
    negative_scores = tf.reduce_max(negative_scores, -1)

    positive_losses = tf.log(1 + tf.exp(gamma * (positive_margin - positive_scores)))
    negative_losses = tf.log(1 + tf.exp(gamma * (negative_margin + negative_scores)))

    losses = positive_losses + negative_losses

    not_easy_mask = tf.greater(losses, 0.)
    not_easy_dists = tf.boolean_mask(losses, not_easy_mask)

    no_pairs = tf.reduce_sum(tf.to_float(not_easy_mask))
    loss = tf.reduce_sum(not_easy_dists) / no_pairs
    loss = tf.cond(tf.equal(no_pairs, 0), lambda: 0., lambda: loss)

    return loss
