import torchvision.transforms as T


def get_default_transform():
    return T.Compose(
        [
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
        ]
    )
