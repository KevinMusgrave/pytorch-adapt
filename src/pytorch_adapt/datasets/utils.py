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


def maybe_download(fn, kwargs):
    original_download = kwargs["download"]
    try:
        kwargs["download"] = False
        fn(**kwargs)
    except (RuntimeError, FileNotFoundError):
        if original_download:
            kwargs["download"] = True
            fn(**kwargs)
        else:
            raise


def num_classes(dataset_name):
    return {
        "mnist": 10,
        "domainnet": 345,
        "domainnet126": 126,
        "office31": 31,
        "officehome": 65,
        "voc_multilabel": 20,
    }[dataset_name]
