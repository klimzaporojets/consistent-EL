import numpy as np
import torch


def collate_character(batch, maxlen, padding, min_word_len=0):
    seqlens = [len(x) for x in batch]
    max_word_len = max([len(w) for sentence in batch for w in sentence])
    maxlen = min(maxlen, max_word_len)
    maxlen = max(maxlen, min_word_len)

    output = torch.LongTensor(len(batch), max(seqlens), maxlen)
    output[:, :, :] = padding
    for i, sentence in enumerate(batch):
        for pos, token in enumerate(sentence):
            token_len = len(token)
            if token_len < maxlen:
                output[i, pos, :len(token)] = torch.from_numpy(np.array(token, dtype=np.long))
            else:
                output[i, pos, :] = torch.from_numpy(np.array(token[0:maxlen], dtype=np.long))
    return output


def decode_table(table):
    table = table.tolist()
    rows = table[0]
    cols = table[1]

    irow = []
    icol = []
    vals = []
    for i in range(rows):
        for j in range(table[2 + i], table[3 + i]):
            irow.append(i)
            icol.append(j)
            vals.append(1.0)

    if len(vals) > 0:
        i = torch.LongTensor([irow, icol])
        v = torch.FloatTensor(vals)
        out = torch.sparse.FloatTensor(i, v, torch.Size([rows, cols]))
        return out
    else:
        return None


def collate_table(batch):
    output = [0, 0]
    rows = 0
    cols = 0

    for table in batch:
        table = table.tolist()
        for i in range(table[0]):
            output.append(table[2 + i] + cols)
        rows += table[0]
        cols += table[1]
    output.append(cols)
    output[0] = rows
    output[1] = cols
    return torch.IntTensor(output)


def collate_sparse(batch):
    m_offs = 0
    c_offs = 0
    irow = []
    icol = []
    vals = []
    for mentions, candidates, rs, cs, vs in batch:
        for r, c, v in zip(rs, cs, vs):
            irow.append(r + m_offs)
            icol.append(c + c_offs)
            vals.append(v)
        m_offs += mentions
        c_offs += candidates

    i = torch.LongTensor([irow, icol])
    v = torch.FloatTensor(vals)
    out = torch.sparse.FloatTensor(i, v, torch.Size([m_offs, c_offs]))
    return out


def collate_mentions_sparse(batch, maxlen):
    rows = len(batch)

    m_offs = 0
    c_offs = 0
    irow = []
    icol = []
    vals = []
    for mentions, _, rs, cs, vs in batch:
        for r, c, v in zip(rs, cs, vs):
            irow.append(r + m_offs)
            icol.append(c + c_offs)
            vals.append(v)
        m_offs += mentions
        c_offs += maxlen

    if m_offs > 0:
        i = torch.LongTensor([irow, icol])
        v = torch.FloatTensor(vals)
        out = torch.sparse.FloatTensor(i, v, torch.Size([m_offs, rows * maxlen]))
        return out
    else:
        return None


def collate_sparse2(batch, num_rows, num_cols):
    m_offs = 0
    c_offs = 0
    irow = []
    icol = []
    vals = []
    for mentions, _, rs, cs, vs in batch:
        for r, c, v in zip(rs, cs, vs):
            irow.append(r + m_offs)
            icol.append(c + c_offs)
            vals.append(v)
        m_offs += num_rows
        c_offs += num_cols

    if m_offs > 0:
        i = torch.LongTensor([irow, icol])
        v = torch.FloatTensor(vals)
        out = torch.sparse.FloatTensor(i, v, torch.Size([m_offs, c_offs]))
        return out
    else:
        return None


def collate_sparse_to_dense_3(batch):
    max_row = max([x[0] for x in batch])
    max_col = max([x[1] for x in batch])
    output = torch.zeros(len(batch), max_row, max_col)
    for b, (rows, cols, r, c, v) in enumerate(batch):
        for x, y, v in zip(r, c, v):
            output[b, x, y] = v
    return output


def collate_sparse_to_dense_4(relations, batch):
    for b, (shape, I, V) in enumerate(batch):
        for s, o, p in I:
            relations[b, s, o, p] = 1.0


def collate_spans(batch):
    span_lengths = [len(x) for x in batch]
    maxlen = max(span_lengths)
    output = []
    for x in batch:
        x = list(x)  # make a copy because we modify it
        if len(x) < maxlen:
            x.extend([[0, 0] for _ in range(maxlen - len(x))])
        output.append(x)
    return span_lengths, torch.LongTensor(output)
