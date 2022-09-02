from torch.hub import load_state_dict_from_url


def download_weights(model, url, pretrained, map_location=None, **kwargs):
    if pretrained:
        state_dict = load_state_dict_from_url(
            url, check_hash=True, map_location=map_location, **kwargs
        )
        if map_location:
            model.to(map_location)
        model.load_state_dict(state_dict)
    return model
