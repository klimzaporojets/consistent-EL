# just to load and debug the input file to hoi model
import json

from modules.bert_preprocessing import flatten

if __name__ == "__main__":
    # hoi_path = 'data/hoi-example/dev.english.384.jsonlines'
    # hoi_path = 'data/hoi/original_examples/dev.english.384_ex4.jsonlines'
    hoi_path = 'data/hoi/output_dwie-1/dwie.train.english.256.jsonlines'

    with open(hoi_path) as infile:
        for curr_line in infile:
            parsed_line = json.loads(curr_line)

            sentences_flattened = flatten(parsed_line['sentences'])

            print('parsed_line.keys(): ', parsed_line.keys())
            print('len(parsed_line[\'tokens\']) = ', len(parsed_line['tokens']))

            print('-----------')
            print('len(parsed_line[\'sentences\']) = ', len(parsed_line['sentences']))
            sum_sentences = 0
            for idx in range(len(parsed_line['sentences'])):
                print('len(parsed_line[\'sentences\'][{}] = '.format(idx), len(parsed_line['sentences'][idx]))
                sum_sentences += len(parsed_line['sentences'][idx])
            print('sum of len of parsed_line[\'sentences\']: ', sum_sentences)

            print('-----------')
            print('len(parsed_line[\'speakers\']) = ', len(parsed_line['speakers']))
            sum_speakers = 0
            for idx in range(len(parsed_line['speakers'])):
                print('len(parsed_line[\'speakers\'][{}] = '.format(idx), len(parsed_line['speakers'][idx]))
                sum_speakers += len(parsed_line['speakers'][idx])
            print('sum of len of parsed_line[\'speakers\']: ', sum_speakers)
            print('-----------')

            print('len(parsed_line[\'clusters\']) = ', len(parsed_line['clusters']))
            print('parsed_line[\'clusters\'] = ', parsed_line['clusters'])
            print('len(parsed_line[\'sentence_map\']) = ', len(parsed_line['sentence_map']))
            print('len(parsed_line[\'subtoken_map\']) = ', len(parsed_line['subtoken_map']))
            print('len(sentences_flattened) = ', len(sentences_flattened))
            print('len(parsed_line[\'constituents\']) = ', len(parsed_line['constituents']))
            print('len(parsed_line[\'ner\']) = ', len(parsed_line['ner']))
            print('len(parsed_line[\'pronouns\']) = ', len(parsed_line['pronouns']))
            print()
