import os


def check_img_paths(img_dir, img_paths, domain):
    for x in img_paths:
        x = os.path.relpath(x, img_dir)
        if not x.startswith(domain):
            raise ValueError(
                f"img_paths contains a path {x} to an image outside of domain {domain}"
            )
