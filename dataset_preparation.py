from os.path import join
import os
import json
import nibabel as nib
import imageio
import shutil
import h5py

import numpy as np
import pydicom

import util


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
    ct = ['ct' in modality.lower() for modality in dataset_json['modality'].values()]
    num_train = len(dataset_json['training'])
    num_test = len(dataset_json['test'])

    for i in range(num_train):
        nib_image = nib.load(join(in_path, dataset_json['training'][i]['image']))
        label = nib.load(join(in_path, dataset_json['training'][i]['label'])).get_fdata()
        np.savez_compressed(join(out_path, f'train_{i}.npz'), image=nib_image.get_fdata(),
                            label=label, spacing=nib_image.header.get_zooms())

    for i in range(num_test):
        nib_image = nib.load(join(in_path, dataset_json['test'][i]))
        np.savez_compressed(join(out_path, f'test_{i}.npz'),
                            image=nib_image.get_fdata(), spacing=nib_image.header.get_zooms())

    add_json_file(out_path, num_train, num_test, ct, len(dataset_json['labels']))


def prepare_ACDC_dataset(in_path, out_path, name):
    dataset_path = join(in_path, name)
    out_path = join(out_path, name)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    test_path = join(dataset_path, 'testing')
    train_path = join(dataset_path, 'training')

    num_train = 0
    for i in range(100):
        idx = str(i+1).zfill(3)
        patient_path = join(train_path, f'patient{idx}')
        patient_files = os.listdir(patient_path)
        for p_file in patient_files:
            if 'frame' in p_file and 'gt' not in p_file and p_file[0] != '.':
                label_name = p_file[0:p_file.index('.')] + '_gt.nii.gz'
                nib_image = nib.load(join(patient_path, p_file))
                label = nib.load(join(patient_path, label_name)).get_fdata()
                np.savez_compressed(join(out_path, f'train_{num_train}.npz'),
                                    image=nib_image.get_fdata(), label=label, spacing=nib_image.header.get_zooms())
                num_train += 1

    num_test = 0
    for i in range(50):
        patient_path = join(test_path, f'patient{101+i}')
        patient_files = os.listdir(patient_path)
        for p_file in patient_files:
            if 'frame' in p_file and p_file[0] != '.':
                nib_image = nib.load(join(patient_path, p_file))
                np.savez_compressed(join(out_path, f'test_{num_test}.npz'),
                                    image=nib_image.get_fdata(), spacing=nib_image.header.get_zooms())
                num_test += 1

    add_json_file(out_path, num_train, num_test, [False], 4)


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
def prepare_CHAOS_dataset(in_path, out_path, name):
    dataset_path = join(in_path, name)
    out_path = join(out_path, name)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    test_path = join(dataset_path, 'Test_Sets', 'MR')
    train_path = join(dataset_path, 'Train_Sets', 'MR')
    files = os.listdir(train_path)
    files.sort()
    num_train = 0
    for patient in files:
        if os.path.isdir(join(train_path, patient)):
            dicom = join(train_path, patient, 'T1DUAL', 'DICOM_anon')
            image1, spacing = dicom2numpy(join(dicom, 'InPhase'))
            image2, _ = dicom2numpy(join(dicom, 'OutPhase'))
            image = np.stack([image1, image2], axis=3)
            label = (png2numpy(join(train_path, patient, 'T1DUAL', 'Ground'))/63).astype(np.uint8)
            np.savez_compressed(join(out_path, f'train_{num_train}.npz'), image=image, label=label, spacing=spacing)

            dicom = join(train_path, patient, 'T2SPIR', 'DICOM_anon')
            image, spacing = dicom2numpy(dicom)
            image = np.stack([image, image], axis=3)
            label = (png2numpy(join(train_path, patient, 'T2SPIR', 'Ground'))/63).astype(np.uint8)
            np.savez_compressed(join(out_path, f'train_{num_train+1}.npz'), image=image, label=label, spacing=spacing)
            num_train += 2

    files = os.listdir(test_path)
    files.sort()
    num_test = 0
    for patient in files:
        if os.path.isdir(join(test_path, patient)):
            dicom = join(test_path, patient, 'T1DUAL', 'DICOM_anon')
            image1, spacing = dicom2numpy(join(dicom, 'InPhase'))
            image2, _ = dicom2numpy(join(dicom, 'OutPhase'))
            image = np.stack([image1, image2], axis=3)
            np.savez_compressed(join(out_path, f'test_{num_test}.npz'), image=image, spacing=spacing)

            dicom = join(test_path, patient, 'T2SPIR', 'DICOM_anon')
            image, spacing = dicom2numpy(dicom)
            image = np.stack([image, image], axis=3)
            np.savez_compressed(join(out_path, f'test_{num_test + 1}.npz'), image=image, spacing=spacing)
            num_test += 2

    add_json_file(out_path, num_train, num_test, [False, False], 5)


def png2numpy(path):
    slices = []
    files = os.listdir(path)
    files.sort()
    for file in files:
        if file[-3:] == 'png':
            slices.append(imageio.imread(join(path, file)))

    return np.stack(slices, axis=2)


def dicom2numpy(path):
    slices = []
    spacings = []
    files = os.listdir(path)
    files.sort() # Ensure they are in ascending order
    for file in files:
        if file[-3:] == 'dcm':
            dcm = pydicom.dcmread(join(path, file))
            slices.append(dcm.pixel_array)
            spacing = list(dcm['PixelSpacing'].value) + [dcm['SpacingBetweenSlices'].value]
            spacings.append([float(i) for i in spacing])

    image = np.stack(slices, axis=2).astype(np.float)
    spacings = np.median(np.array(spacings), axis=0)
    return image, spacings


def prepare_cremi_dataset(in_path, out_path, name):
    dataset_path = join(in_path, name)
    out_path = join(out_path, name)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for idx, letter in enumerate(['A', 'B', 'C']):
        d = h5py.File(join(dataset_path, f'sample_{letter}.hdf'), 'r')
        image = np.array(d['volumes']['raw']).astype(np.float)
        label = np.array(d['volumes']['labels']['clefts'])
        outlier = np.max(label)
        label[label == outlier] = 0
        label[label != 0] = 1
        spacing = d['volumes']['raw'].attrs['resolution']
        np.savez_compressed(join(out_path, f'train_{idx}.npz'), image=image, label=label, spacing=spacing)

        d = h5py.File(join(dataset_path, f'sample_{letter}+.hdf'), 'r')
        image = np.array(d['volumes']['raw']).astype(np.float)
        spacing = d['volumes']['raw'].attrs['resolution']
        np.savez_compressed(join(out_path, f'test_{idx}.npz'), image=image, spacing=spacing)

    add_json_file(out_path, 3, 3, [False], 2)


def add_json_file(path, num_train, num_test, ct, n_classes):
    data = {'num_train': num_train, 'num_test': num_test,
            'ct': ct, 'n_classes': n_classes}
    with open(join(path, 'data.json'), 'w') as outfile:
        json.dump(data, outfile)

