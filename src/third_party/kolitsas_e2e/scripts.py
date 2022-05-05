from third_party.kolitsas_e2e import reader


def create_training_pipelines(experiment_name='corefmerge', all_spans_training=True,
                              datasets=['aida_train'], shuffle_capacity=500,
                              batch_size=1):
    # folder = "../data/kolitsas_e2e/data/tfrecords/" + args.experiment_name + \
    #          ("/allspans/" if args.all_spans_training else "/gmonly/")
    # /home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/data
    # folder = "data/kolitsas_e2e/data/tfrecords/" + experiment_name + \
    #          ("/allspans/" if all_spans_training else "/gmonly/")
    folder = 'data/kolitsas_e2e/data/tfrecords/corefmerge/allspans/'
    training_dataset = reader.train_input_pipeline([folder + file for file in datasets],
                                                   shuffle_capacity=shuffle_capacity, batch_size=batch_size)
    return training_dataset
