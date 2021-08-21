from os.path import join
import os
import json
import nibabel as nib
import dicom2nifti
import dicom2nifti.settings as settings
import shutil

temp_location = "/Volumes/One Touch"
settings.disable_validate_slice_increment()
settings.enable_resampling()
settings.set_resample_spline_interpolation_order(1)
settings.set_resample_padding(-1000)

# TODO: Add JSON file to all dataset preparations, which contains the modality of the channels
# For medical decathlon sets this is found from the json file, or other sets its hard coded


def prepare_decathlon_dataset(in_path, out_path, name):
    """
    Prepares medical decathlon datasets to be in the format expected by the preprocessor.
    :param in_path: Path to the parent directory of the dataset
    :param out_path: Path of output
    :param name: Name of the directory containing the dataset
    """
    in_path = join(in_path, name)
    out_path = join(out_path, name)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    dataset_json = json.load(open(join(in_path, 'dataset.json')))
    ct = ['ct' in modality.lower() for modality in dataset_json['modality']]
    for i in range(dataset_json['numTraining']):
        img_name = dataset_json['training'][i]['image']
        lbl_name = dataset_json['training'][i]['label']
        shutil.copy(join(in_path, img_name), join(out_path, f'trImg_{i}.nii.gz'))
        shutil.copy(join(in_path, lbl_name), join(out_path, f'trLbl_{i}.nii.gz'))

    for i in range(dataset_json['numTest']):
        img_path = join(in_path, dataset_json['test'][i])
        shutil.copy(img_path, join(out_path, f'tsImg_{i}.nii.gz'))

    add_json_file(out_path, dataset_json['numTraining'], dataset_json['numTest'], ct)


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


# This is taking a while, I will return when I have time
def prepare_CHAOS_dataset(path, name):
    dataset_path = join(path, name)
    out_path = join(path, 'temp')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    test_path = join(dataset_path, 'Test_Sets')
    train_pathMR = join(dataset_path, 'Train_Sets', 'MR')
    train_pathCT = join(dataset_path, 'Train_Sets', 'CT')
    mr_files = os.listdir(train_pathMR)
    mr_files.sort()
    index = 0
    for patient in mr_files:
        if patient[0] == '.':
            continue
        dicom = join(train_pathCT, patient, 'T1DUAL', 'DICOM_anon')
        train = dicom2nib(dicom)

        out_phase_files = os.listdir(join(dicom, 'OutPhase'))
        out_phase_files.sort()

        ground_truth_files = os.listdir(join(train_pathCT, patient, 'T1DUAL', 'ground'))
        ground_truth_files.sort()

        index += 1


def dicom2nib(path, single_channel=False):
    if not single_channel:
        folders = os.listdir(path)
        folders.sort()
        # List of each nib file for each channel
        channels = []
        temporary_directories = []
        for channel in folders:
            if channel[0] != '.' and os.path.isdir(join(path, channel)):
                out_path = join(temp_location, channel)
                if not os.path.exists(out_path):
                    os.mkdir(out_path)
                dicom2nifti.convert_directory(join(path, channel), out_path)
                files = [f for f in os.listdir(out_path) if ('.nii.gz' in f)]
                channels.append(nib.load(join(out_path, files[0])))
                temporary_directories.append(out_path)

        result = nib.concat_images(channels)

        for directory in temporary_directories:
            shutil.rmtree(directory)
        return result
    else:
        out_path = join(temp_location, '.temp')
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        dicom2nifti.convert_directory(path, out_path)
        files = [f for f in os.listdir(out_path) if ('.nii.gz' in f)]
        result = nib.load(join(out_path, files[0]))
        shutil.rmtree(out_path)
        return result


def add_json_file(path, num_train, num_test, ct):
    data = {'num_train': num_train, 'num_test': num_test, 'ct': ct}
    with open(join(path, 'data.json'), 'w') as outfile:
        json.dump(data, outfile)


def extract_dataset_signature(path, name):
    pass
