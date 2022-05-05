from modules.ner.spanner import TaskSpan1
# from modules.ner.beam import TaskSpanBeam
from modules.ner.spans import TaskSpanGenerator1, TaskSpanGenerator2
# from modules.tasks.linker import LinkerNone, LossLinker
# from modules.tasks.linkercoref import LossLinkerCoref
from modules.tasks.sequence_labeling import TaskNerSoftmax, TaskNerMulti, TaskCRF, TaskNerRecursive


def create_ner_task(task, dim_inputs, dictionaries):
    task_name, task_type = task['name'], task['type']
    dictionary = dictionaries[task_name]
    if task_type == 'crf':
        return TaskCRF(task_name, dim_inputs, dictionary)
    elif task_type == 'softmax':
        return TaskNerSoftmax(task_name, dim_inputs, dictionary, task)
    elif task_type == 'multi':
        return TaskNerMulti(task_name, dim_inputs, dictionary, task)
    elif task_type == 'recursive':
        return TaskNerRecursive(task_name, dim_inputs, dictionary, task)
    # elif task_type == 'beam': # beam not used according to Johannes
    #     return TaskSpanBeam(task_name, dim_inputs, dictionary, task)
    elif task_type == 'spangen-1':
        return TaskSpanGenerator1(task_name, dim_inputs, task)
    elif task_type == 'spangen-2':
        return TaskSpanGenerator2(task_name, dim_inputs, task)
    elif task_type == 'span-1':
        return TaskSpan1(task_name, dim_inputs, dictionaries[task['dictionary']], task)
    else:
        raise BaseException("unknown task:", task)
