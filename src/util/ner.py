

def decode_segments(indices, labels):
    outputs = []
    for lst in indices:
        data = [labels[x] for x in lst]

        output = []
        start = -1
        type = None
        for pos, target in enumerate(data):
            if target.startswith('B-'):
                if start >= 0:
                    output.append((start, pos, type))
                start = pos
                type = target[2:]
            elif target == 'O':
                if start >= 0:
                    output.append((start, pos, type))
                    start = -1
                    type = None

        if start >= 0:
            output.append((start, len(data), type))
        outputs.append(output)
    return outputs


def ner_to_list(indices, sequence_lengths):
    output = []
    for length, data in zip(sequence_lengths.tolist(), indices.tolist()):
        output.append(data[:length])
    return output


def evaluate_ner_acc(stats, predictions, targets, mask):
    eq = predictions == targets
    correct = eq.float() * mask.float()
    stats['ner_numer'] = stats.get('ner_numer', 0) + correct.sum().item()
    stats['ner_denom'] = stats.get('ner_denom', 0) + mask.float().sum().item()


def update_ner(stats, predictions, targets, labels, obj):
    preds = decode_segments(predictions, labels)
    golds = decode_segments(targets, labels)

    if 'tp' not in stats:
        labels = [x[2:] for x in labels if x.startswith('B-')]
        stats['tp'] = {l: 0 for l in labels}
        stats['fp'] = {l: 0 for l in labels}
        stats['fn'] = {l: 0 for l in labels}

    tp = stats['tp']
    fp = stats['fp']
    fn = stats['fn']

    for pred, gold in zip(preds, golds):
        for _, _, label in [x for x in pred if x in gold]:
            tp[label] += 1
        for _, _, label in [x for x in pred if x not in gold]:
            fp[label] += 1
        for _, _, label in [x for x in gold if x not in pred]:
            fn[label] += 1

    stats['ner-obj'] = stats.get('ner-obj', 0) + obj


def ner_f1(stats):
    print("Evaluate {}".format(stats['name']))
    labels = stats['tp'].keys()
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for label in labels:
        tp, fp, fn = stats['tp'][label], stats['fp'][label], stats['fn'][label]
        pr = tp / (tp + fp) if tp != 0 else 0.0
        re = tp / (tp + fn) if tp != 0 else 0.0
        f1 = 2*tp / (2*tp + fp +fn) if tp != 0 else 0.0
        print('{:24}    {:5}  {:5}  {:5}    {:6.5f}  {:6.5f}  {:6.5f}'.format(label, tp, fp, fn, pr, re, f1))
        total_tp += tp
        total_fp += fp
        total_fn += fn

    total_pr = total_tp / (total_tp + total_fp) if total_tp != 0 else 0.0
    total_re = total_tp / (total_tp + total_fn) if total_tp != 0 else 0.0
    total_f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn) if total_tp != 0 else 0.0
    print('{:24}    {:5}  {:5}  {:5}    {:6.5f}  {:6.5f}  {:6.5f}'.format('', total_tp, total_fp, total_fn, total_pr, total_re, total_f1))
    return total_f1


def evaluate_ner(progress, stats, tb_logger):
    if 'tp' in stats:
        score = ner_f1(stats)
        if 'ner-f1' not in progress:
            progress['ner-f1'] = {'max-value': 0, 'stall': 0}
        p = progress['ner-f1']
        if score >= p['max-value']:
            p['max-value'] = score
            p['max-iter'] = stats['epoch']
            p['stall'] = 0
        else:
            p['stall'] += 1
        p['curr-value'] = score
        p['curr-iter'] = stats['epoch']

        print("EVAL-NER\t{}\tobj: {}\titer: {}\tner-f1: {:6.5f}\tbest-iter: {}\tmax-ner-f1: {:6.5f}\t\tstall: {}"
              .format(stats['name'], stats['ner-obj'], p['curr-iter'], p['curr-value'], p['max-iter'], p['max-value'], p['stall']))
        if tb_logger is not None:
            tb_logger.log_value('{}/f1'.format(stats['name']), score, p['curr-iter'])