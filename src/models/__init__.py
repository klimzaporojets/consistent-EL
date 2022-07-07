# import models.entbydesc as entybydesc
# from models.coref import Coref1
# from models.coref2 import Coref2
from models.models.coreflinker_spanbert_hoi_scorer import CoreflinkerSpanBertHoi
# from models.linker_adv import LinkerAdv1
# from models.lm import LM1, LM2
# from models.more import MyModel6
# from models.ncr import NCR1
# from models.ncr2 import NCR2
# from models.ncr3 import NCR3
# from models.ncr3x import NCR3x
# from models.ncr4 import NCR4
# from models.ncr5 import NCR5
# from models.ncr6 import NCR6
# from models.ner import Ner1, Ner2, Ner3
# from models.ner_link import NerLink1, NerLink2
# from models.relations import Relations1
# from models.relations2 import Relations2
# from models.spanner import SpanNER1
# from models.tc.tc1 import TC1
# from models.test import MyModel3, MyModel4, MyModel5

# from models.linker import Linker1

# from models.linker import MyDygie

models = {}


def register_model(name, factory):
    models[name] = factory


def model_create(config, dictionaries):
    name = config['name']
    if name in models:
        return models[name](dictionaries, config)
    else:
        raise BaseException("no such model:", name)


# register_model('coref-1', Coref1)
# register_model('coref-2', Coref2)
# register_model('entbydecr1', entybydesc.MyModel1)
# register_model('linker_adv_1', Linker1)
# register_model('linker_1', LinkerAdv1)
# register_model('lm_1', LM1)
# register_model('lm_2', LM2)
# register_model('model3', MyModel3)
# register_model('model4', MyModel4)
# register_model('model5', MyModel5)
# register_model('model6', MyModel6)
# register_model('ner_1', Ner1)
# register_model('ner_2', Ner2)
# register_model('ner-3', Ner3)
# register_model('nerlink_1', NerLink1)
# register_model('nerlink_2', NerLink2)
# register_model('relations-1', Relations1)
# register_model('relations-2', Relations2)
# register_model('ncr-1', NCR1)
# register_model('ncr-2', NCR2)
# register_model('ncr-3', NCR3)
# register_model('ncr-3x', NCR3x)
# register_model('ncr-4', NCR4)
# register_model('ncr-5', NCR5)
# register_model('ncr-6', NCR6)
# register_model('spanner-1', SpanNER1)
# register_model('tc-1', TC1)
# register_model('dygie', MyDygie)
# register_model('dygie2', MyDygie2)
# register_model('dygie3', MyDygie3)
# register_model('coreflinker_spanbert', CoreflinkerSpanBert)
register_model('coreflinker_spanbert_hoi', CoreflinkerSpanBertHoi)
