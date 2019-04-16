import sys
import os
import os.path
from lxml import etree
import collections



def create_folder(filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)



def check_entry_dict(event_tokens, d):
    if event_tokens in d:
        return " ".join(d[event_tokens])
    else:
        return event_tokens


def extract_event_CAT(etreeRoot):
    """
    :param etreeRoot: ECB+/ESC XML root
    :return: dictionary with annotaed events in ECB+
    """

    event_dict = collections.defaultdict(list)

    for elem in etreeRoot.findall('Markables/'):
        if elem.tag.startswith("ACTION") or elem.tag.startswith("NEG_ACTION"):
            for token_id in elem.findall('token_anchor'): # the event should have at least one token
                event_mention_id = elem.get('m_id', 'nothing')
                token_mention_id = token_id.get('t_id', 'nothing')
                event_dict[event_mention_id].append(token_mention_id)

    return event_dict


def extract_corefRelations(etreeRoot, d):
    """
    :param etreeRoot: ECB+ XML root
    :return: dictionary with annotaed events in ECB+ (event_dict)
    :return:
    """

    relations_dict_appo = collections.defaultdict(list)
    relations_dict = {}

    for elem in etreeRoot.findall('Relations/'):
        target_element = elem.find('target').get('m_id', 'null') # the target is a non-event
        for source in elem.findall('source'):
            source_elem = source.get('m_id', 'null')
            if source_elem in d:
                val = "_".join(d[source_elem])
                relations_dict_appo[target_element].append(val) # coreferential sets of events

    for k, v in relations_dict_appo.items():
        for i in v:
            relations_dict[i] = v

    return relations_dict


def extract_plotLink(etreeRoot, d):
    """

    :param etreeRoot: ESC XML root
    :param d: dictionary with annotaed events in ESC (event_dict)
    :return:
    """

    plot_dict = collections.defaultdict(list)

    for elem in etreeRoot.findall('Relations/'):
        if elem.tag == "PLOT_LINK":
            source_pl = elem.find('source').get('m_id', 'null')
            target_pl = elem.find('target').get('m_id', 'null')
            relvalu = elem.get('relType', 'null')

            if source_pl in d:
                val1 =  "_".join(d[source_pl])
                if target_pl in d:
                    val2 = "_".join(d[target_pl])
                    plot_dict[(val1, val2)] = relvalu

    return plot_dict


def read_file(ecbplus_original, ecbstart_new, evaluate_file):

    """
    :param ecbplus_original: ECB+ CAT data
    :param ecbstart_new: ESC CAT data
    :param outfile1: event mention extended
    :param outfile2: event extended coref chain
    :return:
    """

    ecbplus = etree.parse(ecbplus_original, etree.XMLParser(remove_blank_text=True))
    root_ecbplus = ecbplus.getroot()
    root_ecbplus.getchildren()

    ecb_event_mentions = extract_event_CAT(root_ecbplus)
    ecb_coref_relations = extract_corefRelations(root_ecbplus, ecb_event_mentions)


    """
    ecbstar data
    """

    ecbstar = etree.parse(ecbstart_new, etree.XMLParser(remove_blank_text=True))
    ecbstar_root = ecbstar.getroot()
    ecbstar_root.getchildren()

    ecb_star_events = extract_event_CAT(ecbstar_root)
    ecbstar_events_plotLink = extract_plotLink(ecbstar_root, ecb_star_events)

    # TLINK ??

    print(ecbplus_original)
    print(ecb_star_events)
    print(ecb_coref_relations)
    print(ecbstar_events_plotLink)



def read_corpus(ecbtopic, ecbstartopic, evaluationtopic):

    """
    :param ecbtopic: ECB+ topic folder in CAT format
    :param ecbstartopic: ESC topic folder in CAT format
    :param outdir: output folder for evaluation data format
    :return:
    """

    if os.path.isdir(ecbtopic) and os.path.isdir(ecbstartopic) and os.path.isdir(evaluationtopic):
        if ecbtopic[-1] != '/':
            ecbtopic += '/'
        if ecbstartopic[-1] != '/':
            ecbstartopic += '/'
        if evaluationtopic[-1] != '/':
            evaluationtopic += '/'

        ecb_subfolder = os.path.dirname(ecbtopic).split("/")[-1]

        for f in os.listdir(ecbtopic):
            if f.endswith('plus.xml'):
                ecb_file = f
                star_file = ecbstartopic + f + ".xml"
                evaluate_file = evaluationtopic + f

                read_file(ecbtopic + ecb_file, star_file, evaluate_file)

            elif f.endswith('ecb.xml'):
                pass
            else:
                print("Missing file" + f)


def main(argv=None):
    ECBplusTopic = '/home/ryan/research/EventStoryLine/ECB+_LREC2014/ECB+/1'
    ECBstarTopic = '/home/ryan/research/EventStoryLine/annotated_data/v0.9/1'
    EvaluationTopic = '/home/ryan/research/EventStoryLine/evaluation_format/full_corpus/v0.9/event_mentions_extended/1'
    read_corpus(ECBplusTopic, ECBstarTopic, EvaluationTopic)


if __name__ == '__main__':
    main()