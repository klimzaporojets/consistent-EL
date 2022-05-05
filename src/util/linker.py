import torch


def update_linker(stats, scores, targets, table, obj):
    predictions = torch.zeros(targets.size()[0], dtype=torch.float)
    table = table.tolist()
    for m in range(table[0]):
        best_score = -1000000
        best_c = -1
        for c in range(table[m + 2], table[m + 3]):
            if scores[c] > best_score:
                best_score = scores[c]
                best_c = c
        predictions[best_c] = 1.0
    correct = torch.mul(predictions, targets.cpu())

    stats['linker_numer'] = stats.get('linker_numer', 0) + torch.sum(correct).item()
    stats['linker_denom'] = stats.get('linker_denom', 0) + torch.sum(targets).item()
    stats['linker-obj'] = stats.get('linker-obj', 0) + obj


def update_linker2(stats, scoresx, targetsx, tables, obj):
    for b, table in enumerate(tables):
        scores, targets = scoresx[b], targetsx[b]

        predictions = torch.zeros(targets.size()[0], dtype=torch.float)
        table = table.tolist()
        for m in range(table[0]):
            best_score = -1000000
            best_c = -1
            for c in range(table[m + 2], table[m + 3]):
                if scores[c] > best_score:
                    best_score = scores[c]
                    best_c = c
            predictions[best_c] = 1.0
        correct = torch.mul(predictions, targets.cpu())
        stats['linker_numer'] = stats.get('linker_numer', 0) + torch.sum(correct).item()
        stats['linker_denom'] = stats.get('linker_denom', 0) + torch.sum(targets).item()

    stats['linker-obj'] = stats.get('linker-obj', 0) + obj


def linker_acc(stats):
    return (stats['linker_numer'] / stats['linker_denom']) if stats['linker_numer'] != 0 else 0


def evaluate_linker(progress, stats):
    if 'linker_numer' in stats:
        score = linker_acc(stats)
        if 'linker-acc' not in progress:
            progress['linker-acc'] = {'max-value': 0, 'stall': 0}
        p = progress['linker-acc']
        if score >= p['max-value']:
            p['max-value'] = score
            p['max-iter'] = stats['epoch']
            p['stall'] = 0
        else:
            p['stall'] += 1
        p['curr-value'] = score
        p['curr-iter'] = stats['epoch']

        print(
            "EVAL-LINKER\t{}\tobj: {}\t{}\tlinker-acc: {} / {} = {}\tmax-iter: {}\tmax-linker-acc: {}\t\tstall: {}"
                .format(stats['name'], stats['linker-obj'], p['curr-iter'], stats['linker_numer'], stats['linker_denom'], score, p['max-iter'], p['max-value'], p['stall']))