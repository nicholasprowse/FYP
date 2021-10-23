def create_examples():
    from util import img2gif, one_hot
    import numpy as np
    import os
    tasks = os.listdir('data/prepared')
    tasks.sort()
    for task in tasks:
        if task[0] == '.':
            continue
        print(task)
        data = np.load(os.path.join('data/prepared', task, 'train_0.npz'))
        image = data['image']
        label = data['label']
        if image.ndim == 3:
            image = np.expand_dims(image, 3)
        for i in range(image.shape[3]):
            img2gif(image[:, :, :, i], 2, f'data/examples/{task}({i}).gif', label=one_hot(label, batch=False))


def create_output_visualisations(task_name, task_id):
    from os.path import join
    import numpy as np
    import json
    from util import img2gif, one_hot
    out_path = f'out/experiment_{task_id}'
    data_path = f'data/prepared/{task_name}'
    config = json.load(open(join(data_path, 'data.json')))
    for i in range(config['num_test']):
        prediction = np.load(join(out_path, f'pred_{i}.npz'))['prediction']
        prediction = one_hot(prediction, n_classes=config['n_classes'], batch=False)
        prediction = prediction.swapaxes(1, 2)
        prediction = prediction.swapaxes(2, 3)
        image = np.load(join(data_path, f'test_{i}.npz'))['image']
        if image.ndim == 3:
            image = np.expand_dims(image, 3)

        for channel in range(image.shape[3]):
            img_channel = image if image.ndim == 3 else image[:, :, :, channel]
            print(img_channel.shape, prediction.shape)
            img2gif(img_channel, 0, join(out_path, f'pred_{i}({channel}).gif'), label=prediction)


def main4():
    import dataset_preparation
    in_path = 'data/raw'
    out_path = 'data/prepared'
    dataset_preparation.prepare_decathlon_dataset(in_path, out_path, 'Task09_Spleen')
    dataset_preparation.prepare_decathlon_dataset(in_path, out_path, 'Task01_BrainTumour')
    dataset_preparation.prepare_decathlon_dataset(in_path, out_path, 'Task02_Heart')
    dataset_preparation.prepare_decathlon_dataset(in_path, out_path, 'Task03_Liver')
    dataset_preparation.prepare_decathlon_dataset(in_path, out_path, 'Task04_Hippocampus')
    dataset_preparation.prepare_decathlon_dataset(in_path, out_path, 'Task05_Prostate')
    dataset_preparation.prepare_decathlon_dataset(in_path, out_path, 'Task06_Lung')
    dataset_preparation.prepare_decathlon_dataset(in_path, out_path, 'Task07_Pancreas')
    dataset_preparation.prepare_decathlon_dataset(in_path, out_path, 'Task08_HepaticVessel')
    dataset_preparation.prepare_decathlon_dataset(in_path, out_path, 'Task10_Colon')


def main5():
    import output
    tasks = ['BrainTumour', 'Heart', 'Liver', 'Hippocampus', 'Prostate', 'Lung', 'Pancreas',
             'HepaticVessel', 'Spleen', 'Colon']
    for i, task in enumerate(tasks):
        idx = f'{i+1}'.zfill(2)
        task_name = f'Task{idx}_{task}'
        output.convert_output_to_nifti(task_name, 401 + i)
        print('Completed', task_name)


if __name__ == '__main__':
    import preprocessing
    preprocessing.obtain_dataset_fingerprint('data/prepared/Task10_Colon', 'data/Task10_Colon', 10 * 1024**2)
    # create_examples()
    # import dataset_preparation
    # dataset_preparation.prepare_cremi_dataset('data/raw', 'data/prepared', 'CREMI')
    # create_output_visualisations('CREMI', 413)
