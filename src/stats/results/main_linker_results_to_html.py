import html
import json
import os

from typing import List, Tuple


class DropdownEntry:
    def __init__(self, type: str, score: float, content: str, span: Tuple[int, int] = None, is_correct=False):
        self.type = type  # coref_mention/ link
        self.score = score
        self.span = span  # (begin,end), probably in char position (the way it appears in gold/prediction files)
        self.content = content
        self.is_correct = is_correct

    def print_entry(self):
        """

        :return: (without the 'div' tag)
    <div class="menu-entry conn_con_1_men_1">20.0 White House (0,1)</div>
    <div class="menu-entry conn_con_1_men_2">15.0 White House (10,11)</div>
    <div class="menu-entry conn_link_2">10.0 <a href="https://en.wikipedia.org/wiki/White_House" target="_blank">White_House</a> (link)</div>
    <div class="menu-entry conn_con_2_men_1">-10.0 Egypt (0,1)</div>

        """
        # to_ret = ''
        if self.type == 'coref_mention':
            # to_ret = '{:.2f} {} {} ({})'.format(self.score, self.content, self.span, self.type)
            to_ret = '{:.2f} {} {}'.format(self.score, self.content, self.span)
        else:
            # it is link
            to_ret = '{:.2f} <a href="https://en.wikipedia.org/wiki/{}" target="_blank">{}</a>' \
                .format(self.score, self.content, self.content)
        return to_ret

    def __str__(self):
        return self.print_entry()


def print_dropdown(dropdown: List[DropdownEntry], concept_id):
    to_ret = ''

    for curr_dropdown in dropdown:
        classes = ['menu-entry']
        if curr_dropdown.type == 'coref_mention':
            conn_class = 'conn_men_{}'.format(get_span_to_str(curr_dropdown.span))
            classes.append(conn_class)
        if curr_dropdown.is_correct:
            classes.append('correct-background')
        else:
            classes.append('incorrect-background')

        to_ret += print_div(classes, curr_dropdown.print_entry())

    return to_ret


def get_span_to_str(span: Tuple[int, int]) -> str:
    return '{}_{}'.format(span[0], span[1])


def print_div(classes: List, content: str):
    to_ret = '<div'
    if len(classes) > 0:
        to_ret += ' class="'
        for idx_class, curr_class in enumerate(classes):
            if idx_class > 0:
                to_ret += ' '
            to_ret += curr_class
        to_ret += '"'
    to_ret += '>'
    to_ret += content
    to_ret += '</div>'
    return to_ret


def add_hover_function_concept(concept_id, background_color='green'):
    """

    :return:

    """
    to_ret = """
    $('div.con_{}').hover(function() {{
     $('div.con_{}').css('background-color', '{}');
   }}, function() {{
     $('div.con_{}').css('background-color', '');
   }});
    """.format(concept_id, concept_id, background_color, concept_id)
    return to_ret


def add_hover_function_conn_prev_mention(concept_id, mention_from_span, mention_to_span, color_highlight='yellow'):
    """

    :return:

    """
    to_ret = """
       $('div.men_{}').hover(function() {{
         $('div.men_{}').css('background-color', '{}');
       }}, function() {{
         $('div.men_{}').css('background-color', '');
       }});    
    """.format(get_span_to_str(mention_from_span), get_span_to_str(mention_to_span),
               color_highlight, get_span_to_str(mention_to_span))
    return to_ret


def add_hover_functions_conn_prev_mention_menu(
        # concept_id:int, span:Tuple[int,int],
        dropdown_entries: List[DropdownEntry], color_highlight='yellow'):
    """

    :return:

    """
    to_ret = ''
    for curr_dropdown_entry in dropdown_entries:
        if 'mention' in curr_dropdown_entry.type:
            to_ret += """
           $('div.conn_men_{}').hover(function() {{
             $('div.men_{}').css('background-color', '{}');
           }}, function() {{
             $('div.men_{}').css('background-color', '');
           }});   
               """.format(get_span_to_str(curr_dropdown_entry.span), get_span_to_str(curr_dropdown_entry.span),
                          color_highlight, get_span_to_str(curr_dropdown_entry.span))
    return to_ret


if __name__ == "__main__":
    experiment_path = 'models/20201022-coreflinker_e2e-ap0-1/'
    output_path = 'models/20201022-coreflinker_e2e-ap0-1/marked_htmls/'
    gold_annotations_path = 'data/data-20200921/'
    id_to_content = dict()

    os.makedirs(output_path, exist_ok=True)
    for (dirpath, dirnames, filenames) in os.walk(gold_annotations_path):
        for filename in filenames:
            if 'DW_' in filename:
                parsed_file = json.load(open(os.path.join(dirpath, filename), encoding='utf-8'), encoding='utf-8')
                doc_id = parsed_file['id']
                doc_content = parsed_file['content']
                id_to_content[doc_id] = {'content': doc_content,
                                         'tokens': parsed_file['tokenization']['tokens'],
                                         'begin': parsed_file['tokenization']['begin'],
                                         'end': parsed_file['tokenization']['end'],
                                         'mentions': parsed_file['mentions'],
                                         'concepts': parsed_file['concepts']}

    in_test_file = open(os.path.join(experiment_path, 'test.jsonl'), encoding='utf-8')
    template_to_fill = open('frontend/templates/template_popup1.html').read()
    for curr_test_line in in_test_file:
        output_file_str = ''
        parsed_line = json.loads(curr_test_line)
        sorted_mentions = sorted(parsed_line['mentions'], key=lambda x: x['begin'])
        added_functions = ''

        doc_id = parsed_line['id']
        content = id_to_content[doc_id]['content']
        output_body = ''
        last_end_tok = 0
        pred_concepts = parsed_line['concepts']
        for curr_pred_concept in pred_concepts:
            added_functions += add_hover_function_concept(curr_pred_concept['concept'])
            added_functions += '\n'

        gold_mentions_pos = {(men['begin'], men['end']) for men in id_to_content[doc_id]['mentions']}
        gold_men_pos_to_concept = {(men['begin'], men['end']): id_to_content[doc_id]['concepts'][men['concept']]
                                   for men in id_to_content[doc_id]['mentions']}

        gold_concept_to_spans = dict()
        for curr_mention in id_to_content[doc_id]['mentions']:
            if curr_mention['concept'] not in gold_concept_to_spans:
                gold_concept_to_spans[curr_mention['concept']] = list()
            gold_concept_to_spans[curr_mention['concept']].append((curr_mention['begin'], curr_mention['end']))

        gold_men_pos_to_other_spans_in_cluster = dict()

        for concept, spans_in_concept in gold_concept_to_spans.items():
            for curr_span in spans_in_concept:
                gold_men_pos_to_other_spans_in_cluster[curr_span] = spans_in_concept

        pred_mentions_pos = {(men['begin'], men['end']) for men in parsed_line['mentions']}
        fp_mention_spans = pred_mentions_pos - gold_mentions_pos
        fn_mention_spans = gold_mentions_pos - pred_mentions_pos
        sorted_fn_spans = sorted(fn_mention_spans, key=lambda x: x[0])

        for begin_tok, end_tok, token in zip(id_to_content[doc_id]['begin'], id_to_content[doc_id]['end'],
                                             id_to_content[doc_id]['tokens']):
            if end_tok <= last_end_tok:
                continue  # already has been printed as a part of multi-word mention

            output_body += html.escape(content[last_end_tok:begin_tok]).encode("ascii", "xmlcharrefreplace").decode('utf-8')
            is_false_positive_mention = False

            while len(sorted_fn_spans) > 0 and begin_tok > sorted_fn_spans[0][1]:
                print('should this happen??, deleting ', begin_tok, sorted_fn_spans[0])
                del sorted_fn_spans[0]

            if len(sorted_mentions) == 0 or begin_tok < sorted_mentions[0]['begin']:

                if len(sorted_fn_spans) > 0 and begin_tok == sorted_fn_spans[0][0]:
                    if end_tok == sorted_fn_spans[0][1]:
                        output_body += print_div(['dropdown', 'false-negative-all', 'men_fn_{}'.format(
                            get_span_to_str(sorted_fn_spans[0]))], html.escape(content[begin_tok:end_tok])
                                                 .encode("ascii", "xmlcharrefreplace").decode('utf-8'))
                        del sorted_fn_spans[0]
                    else:
                        output_body += print_div(['dropdown', 'false-negative-left', 'men_fn_{}'.format(
                            get_span_to_str(sorted_fn_spans[0]))], html.escape(content[begin_tok:end_tok])
                                                 .encode("ascii", "xmlcharrefreplace").decode('utf-8'))
                elif len(sorted_fn_spans) > 0 and begin_tok > sorted_fn_spans[0][0]:
                    if end_tok < sorted_fn_spans[0][1]:
                        output_body += print_div(['dropdown', 'false-negative-middle', 'men_fn_{}'.format(
                            get_span_to_str(sorted_fn_spans[0]))], html.escape(content[begin_tok:end_tok])
                                                 .encode("ascii", "xmlcharrefreplace").decode('utf-8'))
                    elif end_tok == sorted_fn_spans[0][1]:
                        output_body += print_div(['dropdown', 'false-negative-right', 'men_fn_{}'.format(
                            get_span_to_str(sorted_fn_spans[0]))], html.escape(content[begin_tok:end_tok])
                                                 .encode("ascii", "xmlcharrefreplace").decode('utf-8'))
                        del sorted_fn_spans[0]
                    else:
                        print('Should not happen for end_tok to be larger than sorted_fn_spans[0][1]!!',
                              end_tok, sorted_fn_spans[0][1])
                        del sorted_fn_spans[0]
                else:
                    output_body += html.escape(content[begin_tok:end_tok]).encode("ascii", "xmlcharrefreplace").decode(
                        'utf-8')

                if last_end_tok < end_tok:
                    last_end_tok = end_tok
            else:
                curr_mention = sorted_mentions[0]
                if begin_tok > sorted_mentions[0]['begin']:
                    # some mentions are overlapped, so print only one and ignore the rest ; if not one is printed
                    # next to the other producing incorrect output
                    del sorted_mentions[0]
                    continue

                if (curr_mention['begin'], curr_mention['end']) in fp_mention_spans:
                    is_false_positive_mention = True

                mention_tok_begin = curr_mention['begin']
                output_body += html.escape(content[begin_tok:mention_tok_begin]).encode("ascii",
                                                                                        "xmlcharrefreplace").decode(
                    'utf-8')
                mention_concept = curr_mention['concept']
                mention_span = (curr_mention['begin'], curr_mention['end'])
                mention_text = html.escape(curr_mention['text']).encode("ascii", "xmlcharrefreplace").decode('utf-8')

                if 'coref_connection_type' in curr_mention:
                    if 'mention' in curr_mention['coref_connection_type']:
                        mention_other_span = (curr_mention['coref_connection_pointer']['begin'],
                                              curr_mention['coref_connection_pointer']['end'])

                        added_functions += add_hover_function_conn_prev_mention(mention_concept,
                                                                                mention_span,
                                                                                mention_other_span)
                else:
                    print('WARNING!! - {} coref_connection_type not in '.format(doc_id), curr_mention)

                # builds the dropdown
                dt_entries = list()
                gold_concept = gold_men_pos_to_concept.get((curr_mention['begin'], curr_mention['end']))
                gold_other_mentions_spans = None  # TODO!!
                if 'candidates' in curr_mention:
                    for curr_link_candidate, curr_link_score in zip(curr_mention['candidates'], curr_mention['scores']):
                        if not gold_concept is None and 'link' in gold_concept:
                            dt_entries.append(DropdownEntry('link', curr_link_score, curr_link_candidate, None,
                                                            curr_link_candidate == gold_concept['link']))
                        else:
                            dt_entries.append(DropdownEntry('link', curr_link_score, curr_link_candidate, None))

                if 'coref_scores' in curr_mention:
                    for curr_coref_score in curr_mention['coref_scores']:
                        correct_spans = set()

                        if (curr_mention['begin'], curr_mention['end']) in gold_men_pos_to_other_spans_in_cluster:
                            is_correct = tuple(curr_coref_score['span']) \
                                         in gold_men_pos_to_other_spans_in_cluster.get(
                                (curr_mention['begin'], curr_mention['end']))

                            dt_entries.append(DropdownEntry('coref_mention', curr_coref_score['score'],
                                                            html.escape(curr_coref_score['text']).encode("ascii",
                                                                                                         "xmlcharrefreplace").decode(
                                                                'utf-8'),
                                                            tuple(curr_coref_score['span']),
                                                            is_correct=is_correct))
                        else:
                            dt_entries.append(DropdownEntry('coref_mention', curr_coref_score['score'],
                                                            html.escape(curr_coref_score['text']).encode("ascii",
                                                                                                         "xmlcharrefreplace").decode(
                                                                'utf-8'),
                                                            tuple(curr_coref_score['span'])))
                else:
                    print('WARNING!! - {} coref_scores not in '.format(doc_id), curr_mention)

                # sorts by scores in descending order
                dt_entries = sorted(dt_entries, key=lambda x: x.score, reverse=True)
                added_functions += add_hover_functions_conn_prev_mention_menu(dt_entries)
                dropdown_content = print_dropdown(dt_entries, mention_concept)

                if 'link_pred' in curr_mention:
                    mention_text = '<a href="https://en.wikipedia.org/wiki/{}" target="_blank">{}</a>' \
                        .format(curr_mention['link_pred'], mention_text)

                if not is_false_positive_mention:
                    output_body += print_div(['dropdown', 'green-background', 'con_{}'.format(mention_concept),
                                              'men_{}'.format(get_span_to_str(mention_span))],
                                             mention_text + print_div(['dropdown-content'], dropdown_content))
                else:
                    output_body += print_div(['dropdown', 'false-positive', 'con_{}'.format(mention_concept),
                                              'men_{}'.format(get_span_to_str(mention_span))],
                                             mention_text + print_div(['dropdown-content'], dropdown_content))

                last_end_tok = sorted_mentions[0]['end']
                del sorted_mentions[0]

        output_file_str = template_to_fill.replace('##ADDED_FUNCTIONS##', added_functions)
        output_file_str = output_file_str.replace('##ADDED_BODY##', output_body)

        output_file_path = '{}.html'.format(doc_id)
        output_file_path = os.path.join(output_path, output_file_path)

        output_file = open(output_file_path, 'w', encoding='utf-8')
        output_file.write(output_file_str)
        output_file.close()

    print('len of id_to_content: ', len(id_to_content))
