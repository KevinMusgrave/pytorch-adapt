import torch

from ..utils import common_functions as c_f


# reference: https://github.com/wgchang/DSBN/blob/master/model/dsbn.py
class AdaptiveBatchNorm(torch.nn.Module):
    def __init__(
        self, num_features=None, num_domains=2, affine_domain=None, bns=None, **kwargs
    ):
        super().__init__()
        if affine_domain is not None and not isinstance(affine_domain, int):
            raise TypeError("affine_domain must either be None or an int")
        if bns is None:
            bns = []
            for i in range(num_domains):
                bns.append(self.get_batchnorm_class()(num_features, **kwargs))
        self.bns = torch.nn.ModuleList(bns)
        self.affine_domain = affine_domain
        self.curr_domain = None
        self.set_affine_params(delete=True)

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def set_curr_domain(self, domain):
        self.curr_domain = domain

    def set_affine_params(self, delete=False):
        if self.affine_domain is not None:
            source_bn = self.bns[self.affine_domain]
            weight = source_bn.weight.data
            bias = source_bn.bias.data
            for i in range(len(self.bns)):
                if i != self.affine_domain:
                    target_bn = self.bns[i]
                    if delete:
                        del target_bn.weight
                        del target_bn.bias
                    target_bn.weight = weight
                    target_bn.bias = bias

    def forward(self, x, domain=None):
        domain = c_f.default(domain, self.curr_domain)
        if domain is None:
            raise TypeError(
                "If input domain is None, then curr_domain must be set via set_curr_domain"
            )
        self.set_affine_params(delete=False)
        output = self.bns[domain](x)
        self.set_curr_domain(None)
        return output


class AdaptiveBatchNorm2d(AdaptiveBatchNorm):
    def get_batchnorm_class(self):
        return torch.nn.BatchNorm2d


# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
# https://stackoverflow.com/a/56407442
def update(existingAggregate, newValues):
    newValues = newValues.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2)
    (count, mean, M2) = existingAggregate
    count += newValues.shape[0]
    # newvalues - oldMean
    delta = newValues - mean
    mean += torch.sum(delta / count, dim=0)
    # newvalues - newMean
    delta2 = newValues - mean
    M2 += torch.sum(delta * delta2, dim=0)
    return (count, mean, M2)


def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
    if count < 2:
        raise ValueError("Can't finalize when count is less than 2")
    else:
        return (mean, variance, sampleVariance)


# original https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
class PopulationBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(
        self, num_features, eps=1e-5, affine=True, init_mean=None, init_var=None
    ):
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=None,
            affine=affine,
            track_running_stats=True,
        )
        self.total_n = 0
        self.register_buffer("M2", torch.zeros(num_features))
        self.register_buffer(
            "final_mean", c_f.default(init_mean, torch.zeros(num_features))
        )
        self.register_buffer(
            "final_var", c_f.default(init_var, torch.ones(num_features))
        )

    def finalize(self):
        try:
            self.final_mean, self.final_var, _ = finalize(
                (self.total_n, self.running_mean, self.M2)
            )
        except ValueError:
            pass

    def reset_running_stats(self):
        super().reset_running_stats()
        self.total_n = 0
        for attr in ["M2", "final_mean", "final_var"]:
            if hasattr(self, attr):
                if attr == "final_var":
                    getattr(self, attr).fill_(1)
                else:
                    getattr(self, attr).zero_()

    def forward(self, input):
        self._check_input_dim(input)
        # calculate running estimates
        if self.training:
            with torch.no_grad():
                self.total_n, self.running_mean, self.M2 = update(
                    (self.total_n, self.running_mean, self.M2), input
                )

        mean = self.final_mean
        var = self.final_var

        input = (input - mean[None, :, None, None]) / (
            torch.sqrt(var[None, :, None, None] + self.eps)
        )
        if self.affine:
            input = (
                input * self.weight[None, :, None, None]
                + self.bias[None, :, None, None]
            )

        return input


def convert_bn_to_adabn(model, affine_domain, bn_type):
    changes = []
    for child_name, child in model.named_children():
        if isinstance(child, (torch.nn.BatchNorm1d, torch.nn.BatchNorm3d)):
            raise TypeError(
                f"child is type {type(child)}, which is not currently supported for conversion to adabn."
            )
        if isinstance(child, torch.nn.BatchNorm2d):
            num_features = child.num_features
            if bn_type == PopulationBatchNorm2d:
                other_bn = bn_type(
                    num_features,
                    init_mean=child.running_mean.clone(),
                    init_var=child.running_var.clone(),
                )
            elif bn_type == torch.nn.BatchNorm2d:
                other_bn = bn_type(num_features)
            new_bn = AdaptiveBatchNorm2d(
                affine_domain=affine_domain, bns=[child, other_bn]
            )
            setattr(model, child_name, new_bn)
        convert_bn_to_adabn(child, affine_domain, bn_type)


def collect_all_bn(model, bn_type):
    x = []
    for m in model.children():
        if isinstance(m, bn_type):
            x.append(m)
        x.extend(collect_all_bn(m, bn_type))
    return x


def collect_all_bn_if_not_done(model, bn_type):
    list_name = f"list_of_{bn_type.__name__}"
    if not hasattr(model, list_name):
        x = collect_all_bn(model, bn_type)
        setattr(model, list_name, x)
    return list_name


def set_curr_domain(model, domain, bn_type):
    list_name = collect_all_bn_if_not_done(model, bn_type)
    for v in getattr(model, list_name):
        v.set_curr_domain(domain)


def finalize_bn(model, bn_type):
    list_name = collect_all_bn_if_not_done(model, bn_type)
    for v in getattr(model, list_name):
        v.finalize()


def set_bn_layer_to_train(model, layer_num, bn_type):
    list_name = collect_all_bn_if_not_done(model, bn_type)
    list_of_bn = getattr(model, list_name)
    set_layer = None
    for i, v in enumerate(list_of_bn):
        if i == layer_num:
            v.train()
            set_layer = i
        else:
            v.eval()
    c_f.LOGGER.info(f"There are {len(list_of_bn)} bn layers")
    if set_layer is not None:
        c_f.LOGGER.info(f"Set bn layer {set_layer} to train mode")
    else:
        c_f.LOGGER.info(f"Did not set any bn layers to train mode.")
