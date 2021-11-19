from torch.hub import load_state_dict_from_url


def download_weights(model, url, pretrained, progress):
    if pretrained:
        state_dict = load_state_dict_from_url(url, progress=progress)
        model.load_state_dict(state_dict)
    return model
