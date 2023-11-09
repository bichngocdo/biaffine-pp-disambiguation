import argparse


class CoNLLFile:
    def __init__(self, f):
        self.file = f
        self.sentence = list()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            line = self.file.readline()
            if not line:
                if not self.sentence:
                    raise StopIteration
                else:
                    sentences = self.sentence
                    self.sentence = []
                    return sentences
            else:
                line = line.rstrip()
                if not line and self.sentence:
                    sentences = self.sentence
                    self.sentence = []
                    return sentences
                if line:
                    self.sentence.append(line)

    def close(self):
        self.file.close()


PPS = {'APPR', 'APPRART', 'APPO'}


def _is_N_V(tag):
    return tag[0] in {'N', 'V'}


def eval(fp, only_nv=True):
    with open(fp, 'r') as f:
        f = CoNLLFile(f)
        no_gold = 0
        no_correct = 0
        no_incorrect = 0

        acc_no_correct = 0

        for block in f:
            block = [line.split('\t') for line in block]
            for parts in block:
                gold_head_pos = block[int(parts[6]) - 1][3]

                if parts[3] in PPS and (not only_nv or _is_N_V(gold_head_pos)):
                    no_gold += 1
                    if parts[6] == parts[8]:
                        acc_no_correct += 1
                    if parts[4] in PPS:
                        if parts[6] == parts[8]:
                            no_correct += 1
                        else:
                            no_incorrect += 1

        no_pred = no_correct + no_incorrect

        a = 100. * acc_no_correct / no_gold
        p = 100. * no_correct / no_pred
        r = 100. * no_correct / no_gold
        f1 = 2 / (1 / p + 1 / r)
        print('A : %d / %d = %.2f' % (acc_no_correct, no_gold, a))
        print('P : %d / %d = %.2f' % (no_correct, no_pred, p))
        print('R : %d / %d = %.2f' % (no_correct, no_gold, r))
        print('F1: %.2f' % f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PP disambiguation evaluation for CoNLL 2009 format')
    parser.add_argument('input',
                        help='Input file in CoNLL 2009 format. '
                             'Column 9 and 10 contain the gold and predicted dependency heads, '
                             'column 11 and 12 contain the gold and predicted dependency labels.')
    parser.add_argument('--only_nv', action='store_true',
                        help='Evaluate only on prepositions that have nouns or verbs as objects')
    args = parser.parse_args()

    eval(args.input, args.only_nv)
