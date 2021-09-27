import numpy as np
from scipy.ndimage import label


def remove_all_but_the_largest_connected_component(image, n_classes):
    """
    removes all but the largest connected component, individually for each class
    :param image: Numpy array
    :return:
    Adapted from nnUnet
    """
    largest_removed = {}
    kept_size = {}
    for c in range(1, n_classes):
        if isinstance(c, (list, tuple)):
            c = tuple(c)  # otherwise it cant be used as key in the dict
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int))

        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum()

        largest_removed[c] = None
        kept_size[c] = None

        if num_objects > 0:
            # we always keep the largest object. We could also consider removing the largest object if it is smaller
            # than minimum_valid_object_size in the future but we don't do that now.
            maximum_size = max(object_sizes.values())
            kept_size[c] = maximum_size

            for object_id in range(1, num_objects + 1):
                # we only remove objects that are not the largest
                if object_sizes[object_id] != maximum_size:
                    image[(lmap == object_id) & mask] = 0
                    if largest_removed[c] is None:
                        largest_removed[c] = object_sizes[object_id]
                    else:
                        largest_removed[c] = max(largest_removed[c], object_sizes[object_id])
    return image
