import os


def check_img_paths(img_dir, img_paths, domain):
    for x in img_paths:
        x = os.path.relpath(x, img_dir)
        if not x.startswith(domain):
            raise ValueError(
                f"img_paths contains a path {x} to an image outside of domain {domain}"
            )


def check_length(cls, correct_length):
    x = len(cls)
    if x != correct_length:
        raise ValueError(f"len(self)={x} but should be {correct_length}")


def check_train(train):
    if not isinstance(train, bool):
        raise TypeError("train should be True or False")
    return train
