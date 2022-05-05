## just for debugging purposes
from third_party.kolitsas_e2e.scripts import create_training_pipelines

if __name__ == "__main__":
    print("Start")
    pipeline_data = create_training_pipelines()

    data_iterator = pipeline_data.take(50)
    for sample in data_iterator:
        print('current iterated sample: ', sample)
        # image, label = sample["image"].numpy(), sample["label"].numpy()
        # plt.imshow(image[:, :, 0].astype(np.uint8), cmap=plt.get_cmap("gray"))
        # plt.show()
        # print("Label: %d" % label)
    print('pipeline_data: ', pipeline_data)
