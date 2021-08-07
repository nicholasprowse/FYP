from os.path import join
import os
import json
import shutil


def prepare_decathlon_dataset(path, name):
    """
    Prepares medical decathlon datasets to be in the format expected by the preprocessor.
    :param path: Path to the parent directory of the dataset
    :param name: Name of the directory containing the dataset
    """
    dataset_path = join(path, name)
    out_path = join(path, 'temp')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    dataset_json = json.load(open(join(dataset_path, 'dataset.json')))
    for i in range(dataset_json['numTraining']):
        img_name = dataset_json['training'][i]['image']
        lbl_name = dataset_json['training'][i]['label']
        shutil.move(join(dataset_path, img_name), join(out_path, f'trImg_{i}.nii.gz'))
        shutil.move(join(dataset_path, lbl_name), join(out_path, f'trLbl_{i}.nii.gz'))

    for i in range(dataset_json['numTest']):
        img_path = join(dataset_path, dataset_json['test'][i])
        shutil.move(img_path, join(out_path, f'tsImg_{i}.nii.gz'))

    shutil.rmtree(dataset_path, ignore_errors=True)
    os.mkdir(dataset_path)
    os.replace(out_path, dataset_path)


def prepare_ACDC_dataset(path, name):
    dataset_path = join(path, name)
    out_path = join(path, 'temp')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    test_path = join(dataset_path, 'testing')
    train_path = join(dataset_path, 'training')
    training_files = os.listdir(train_path)
    # Sort to ensure the files in the train and test set are in the same order
    training_files.sort()
    index = 0
    for t_file in training_files:
        patient_path = join(train_path, t_file)
        if os.path.isdir(patient_path):
            patient_files = os.listdir(patient_path)
            for p_file in patient_files:
                if 'frame' in p_file and 'gt' not in p_file and p_file[0] != '.':
                    label_name = p_file[0:p_file.index('.')] + '_gt.nii.gz'
                    shutil.move(join(patient_path, p_file), join(out_path, f'trImg_{index}.nii.gz'))
                    shutil.move(join(patient_path, label_name), join(out_path, f'trLbl_{index}.nii.gz'))
                    index += 1

    index = 0
    testing_file = os.listdir(test_path)
    testing_file.sort()
    for t_file in testing_file:
        patient_path = join(test_path, t_file)
        if os.path.isdir(patient_path):
            patient_files = os.listdir(patient_path)
            for p_file in patient_files:
                if 'frame' in p_file and p_file[0] != '.':
                    shutil.move(join(patient_path, p_file), join(out_path, f'tsImg_{index}.nii.gz'))
                    index += 1

    shutil.rmtree(dataset_path, ignore_errors=True)
    os.mkdir(dataset_path)
    os.replace(out_path, dataset_path)


def prepare_synapse_dataset(path, name):
    dataset_path = join(path, name)
    out_path = join(path, 'temp')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    test_path = join(dataset_path, 'testing')
    train_path = join(dataset_path, 'training')
    label_path = join(dataset_path, 'training_labels')
    training_files = os.listdir(train_path)
    training_files.sort()
    index = 0
    for t_file in training_files:
        if t_file[0] != '.':
            label_name = t_file[0:t_file.index('.')] + "_seg.nii.gz"
            shutil.move(join(train_path, t_file), join(out_path, f'trImg_{index}.nii.gz'))
            shutil.move(join(label_path, label_name), join(out_path, f'trLbl_{index}.nii.gz'))
            index += 1

    testing_files = os.listdir(test_path)
    testing_files.sort()
    index = 0
    for t_file in testing_files:
        if t_file[0] != '.':
            shutil.move(join(test_path, t_file), join(out_path, f'tsImg_{index}.nii.gz'))
            index += 1

    shutil.rmtree(dataset_path, ignore_errors=True)
    os.mkdir(dataset_path)
    os.replace(out_path, dataset_path)


def prepare_CHAOS_dataset(path, name):
    dataset_path = join(path, name)
    out_path = join(path, 'temp')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    test_path = join(dataset_path, 'Test_Sets')
    train_pathMR = join(dataset_path, 'Train_Sets/MR')
    train_pathCT = join(dataset_path, 'Train_Sets/CT')
    ct_files = os.listdir(train_pathCT)
    ct_files.sort()
    index = 0
    for t_file in ct_files:
        dicom = join(train_pathCT, t_file, 'T1DUAL', 'DICOM_anon')
        in_phase_files = os.listdir(join(dicom, 'InPhase'))
        in_phase_files.sort()


        index += 1





def extract_dataset_signature(path, name):
    pass
