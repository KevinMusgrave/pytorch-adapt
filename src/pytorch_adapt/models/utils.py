from torch.hub import load_state_dict_from_url


def download_weights(model, url, pretrained, progress, file_name):
    if pretrained:
        state_dict = load_state_dict_from_url(
            url, progress=progress, file_name=file_name
        )
        model.load_state_dict(state_dict)
    return model
