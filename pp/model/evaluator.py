from collections import OrderedDict


class Stats(object):
    def __init__(self, name):
        self.name = name

        self.loss = 0.
        self.time = 0.
        self.num_iterations = 0
        self.num_instances = 0
        self.num_sentences = 0
        self.num_correct_instances = 0
        self.num_correct_sentences = 0

    def reset(self):
        self.loss = 0.
        self.time = 0.
        self.num_iterations = 0
        self.num_instances = 0
        self.num_sentences = 0
        self.num_correct_instances = 0
        self.num_correct_sentences = 0

    def update(self, loss, time_elapsed, gold, prediction):
        self.loss += loss
        self.time += time_elapsed
        self.num_iterations += 1

        for gold_sentence, predicted_sentence in zip(gold, prediction):
            num_errors = 0
            num_instances = 0
            for gold_label, predicted_label in zip(gold_sentence, predicted_sentence):
                if predicted_label >= 0:
                    num_instances += 1
                    if gold_label == predicted_label:
                        self.num_correct_instances += 1
                    else:
                        num_errors += 1

            self.num_instances += num_instances
            self.num_sentences += num_instances > 0
            self.num_correct_sentences += num_errors == 0 and num_instances > 0

    def aggregate(self):
        results = OrderedDict()
        results['%s_loss' % self.name] = self.loss / self.num_iterations
        results['%s_rate' % self.name] = self.num_sentences / self.time
        results['%s_acc' % self.name] = self.num_correct_instances / self.num_instances \
            if self.num_instances > 0 else float('NaN')
        results['%s_sent_acc' % self.name] = self.num_correct_sentences / self.num_sentences \
            if self.num_sentences > 0 else float('NaN')
        self.reset()
        return results
