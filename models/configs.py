
Split_Configs = {
    'mlp2': {
        1: [],  # End-to-end
        2: [0],
        3: [0, 1],
    },
    'mlp3': {
        1: [],  # End-to-end
        2: [1],
        3: [0, 1],
        4: [0, 1, 2],
    },
    'resnet18': { # 0-base conv, 1-2, 3-4, 5-6, 7-8, 9: Avg-linear
        1: [],  # End-to-end
        2: [4],
        3: [2, 6],
        4: [2, 4, 6],
        8: [2, 3, 4, 5, 6, 7, 8],
    },
    'resnet34': { # 0-base conv, 1-3, 4-7, 8-13, 14-16, 17: Avg-linear
        1: [],  # End-to-end
        2: [8],
        3: [5, 12],
        4: [4, 8, 12],
        8: [2, 4, 6, 8, 10, 12, 14],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    },
    'resnet50': { # 0-base conv, 1-3, 4-7, 8-13, 14-16, 17: Avg-linear
        1: [],  # End-to-end
        2: [8],
        3: [5, 12],
        4: [4, 8, 12],
        8: [2, 4, 6, 8, 10, 12, 14],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    },
}

EXNN_Split_Configs = {
    'mlp2': {
        1: [],  # End-to-end
        2: [0],
        3: [0, 1],
    },
    'mlp3': {
        1: [],  # End-to-end
        2: [1],
        3: [0, 1],
        4: [0, 1, 2],
    },
    'resnet18': { # 0-base conv, 1-2, 3-4, 5-6, 7-8, 9: Avg-linear
        1: [],  # End-to-end
        2: [4],
        3: [0, 4],
        4: [0, 4, 6],
        8: [0, 1, 2, 3, 5, 7, 8],
    },
}


InfoPro = {
    'mlp2': {
        1: [],  # End-to-end
        2: [0],
        3: [0, 1],
    },
    'mlp3': {
        1: [],  # End-to-end
        2: [1],
        3: [0, 1],
        4: [0, 1, 2],
    },
    'resnet18': { # 0-base conv, 1-2, 3-4, 5-6, 7-8, 9: Avg-linear
        1: [],  # End-to-end
        2: [4],
        3: [2, 6],
        4: [2, 4, 6],
        8: [2, 3, 4, 5, 6, 7, 8],
    },
    'resnet34': { # 0-base conv, 1-3, 4-7, 8-13, 14-16, 17: Avg-linear
        1: [],  # End-to-end
        2: [8],
        3: [5, 12],
        4: [4, 8, 12],
        8: [2, 4, 6, 8, 10, 12, 14],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    },
    'resnet50': { # 0-base conv, 1-3, 4-7, 8-13, 14-16, 17: Avg-linear
        1: [],  # End-to-end
        2: [8],
        3: [5, 12],
        4: [4, 8, 12],
        8: [2, 4, 6, 8, 10, 12, 14],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    },
}


InfoPro_balanced_memory = {
    'mlp2': {
        1: [],  # End-to-end
        2: [0],
        3: [0, 1],
    },
    'mlp3': {
        1: [],  # End-to-end
        2: [1],
        3: [0, 1],
        4: [0, 1, 2],
    },
    'resnet18': { # 0-base conv, 1-2, 3-4, 5-6, 7-8, 9: Avg-linear
        1: [],  # End-to-end
        2: [4],
        3: [2, 6],
        4: [2, 4, 6],
        8: [2, 3, 4, 5, 6, 7, 8],
    },
    'resnet34': { # 0-base conv, 1-3, 4-7, 8-13, 14-16, 17: Avg-linear
        1: [],  # End-to-end
        2: [8],
        3: [5, 12],
        4: [4, 8, 12],
        8: [2, 4, 6, 8, 10, 12, 14],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    },
    'resnet50': { # 0-base conv, 1-3, 4-7, 8-13, 14-16, 17: Avg-linear
        1: [],  # End-to-end
        2: [8],
        3: [5, 12],
        4: [4, 8, 12],
        8: [2, 4, 6, 8, 10, 12, 14],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    },
}
