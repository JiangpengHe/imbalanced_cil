from os.path import join

_BASE_DATA_PATH = "../data"

dataset_config = {
    'cifar100': {
        'path': join(_BASE_DATA_PATH, 'cifar100'),
        'resize': None,
        'pad': 4,
        'crop': 32,
        'flip': True,
        'normalize': ((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
    },
    'cifar100_conv': {
        'path': join(_BASE_DATA_PATH, 'cifar100'),
        'resize': None,
        'pad': 4,
        'crop': 32,
        'flip': True,
        'normalize': ((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        'class_order': [
            68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
            28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
            98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
            36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33
        ]
    },
    'cifar100_lt': {
        'path': join(_BASE_DATA_PATH, 'cifar100'),
        'resize': None,
        'pad': 4,
        'crop': 32,
        'flip': True,
        'normalize': ((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        'class_order': [
            68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
            28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
            98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
            36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33
        ]
    },
    'cifar100_ltio': {
        'path': join(_BASE_DATA_PATH, 'cifar100'),
        'resize': None,
        'pad': 4,
        'crop': 32,
        'flip': True,
        'normalize': ((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        'class_order': [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 
            57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
             84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99
        ]
    },
    'imagenet_subset_conv': {
        'path': join(_BASE_DATA_PATH, 'ImageNetSubset'),
        'resize': None,
        'crop': 224,
        'flip': True,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        'class_order': [
            68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
            28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
            98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
            36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33
        ]
    },
    'imagenet_subset_lt': {
        'path': join(_BASE_DATA_PATH, 'ImageNetSubset'),
        'resize': None,
        'crop': 224,
        'flip': True,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        'class_order': [
            68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
            28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
            98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
            36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33
        ]
    },
    'imagenet_subset_ltio': {
        'path': join(_BASE_DATA_PATH, 'ImageNetSubset'),
        'resize': None,
        'crop': 224,
        'flip': True,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        'class_order': [
            68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
            28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
            98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
            36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33
        ]
    },
    'Food101_lt': {
        'path': join(_BASE_DATA_PATH, 'Food101'),
        'resize': 224,
        'crop': 224,
        'flip': True,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    },

    'Food101_ltio': {
        'path': join(_BASE_DATA_PATH, 'Food101'),
        'resize': 224,
        'crop': 224,
        'flip': True,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    },

    'imagenet_1000_lt': {
        'path': join(_BASE_DATA_PATH, 'imagenet_1000'),
        'resize': 224,
        'crop': 224,
        'flip': True,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    }

}

for dset in dataset_config.keys():
    for k in ['resize', 'pad', 'crop', 'normalize', 'class_order', 'extend_channel']:
        if k not in dataset_config[dset].keys():
            dataset_config[dset][k] = None
    if 'flip' not in dataset_config[dset].keys():
        dataset_config[dset]['flip'] = False