import copy
import unittest

import torch

from pytorch_adapt.adapters import (
    ADDA,
    CDAN,
    DANN,
    GAN,
    GVB,
    MCD,
    RTN,
    VADA,
    AdaBN,
    Aligner,
    Classifier,
    DomainConfusion,
    Finetuner,
    SymNets,
)
from pytorch_adapt.containers import Misc
from pytorch_adapt.hooks import (
    AdaBNHook,
    ADDAHook,
    AlignerPlusCHook,
    CDANHook,
    ClassifierHook,
    DANNHook,
    DomainConfusionHook,
    FinetunerHook,
    GANHook,
    GVBHook,
    JointAlignerHook,
    MCDHook,
    RTNHook,
    SymNetsHook,
    VADAHook,
)
from pytorch_adapt.layers import (
    AdaBNModel,
    CORALLoss,
    MMDLoss,
    MultipleModels,
    PlusResidual,
    RandomizedDotProduct,
)
from pytorch_adapt.models import Discriminator
from pytorch_adapt.utils import common_functions as c_f
from pytorch_adapt.weighters import MeanWeighter

from .. import TEST_FOLDER
from .run_adapter import run_adapter
from .utils import get_gcd


def common_log_files():
    return {
        "validator_WithHistory": {
            "latest_score",
            "best_score",
            "latest_epoch",
            "best_epoch",
        },
        "stat_getter_WithHistory": {
            "latest_score",
            "best_score",
            "latest_epoch",
            "best_epoch",
        },
    }


def gan_log_files():
    log_files = common_log_files()
    log_files.update(
        {
            "optimizers_G_Adam": {"lr"},
            "optimizers_C_Adam": {"lr"},
            "optimizers_D_Adam": {"lr"},
            "engine_output_g_loss": {
                "total",
                "g_src_domain_loss",
                "g_target_domain_loss",
                "c_loss",
            },
            "engine_output_d_loss": {
                "total",
                "d_src_domain_loss",
                "d_target_domain_loss",
            },
        }
    )
    return log_files


class TestRunning(unittest.TestCase):
    def test_adabn(self):
        models = get_gcd()
        del models["D"]
        for k, v in models.items():
            models[k] = AdaBNModel(v)
        adapter = AdaBN(models=models)
        self.assertTrue(isinstance(adapter.hook, AdaBNHook))
        run_adapter(self, TEST_FOLDER, adapter, common_log_files())

    def test_adda(self):
        adapter = ADDA(models=get_gcd())
        self.assertTrue(isinstance(adapter.hook, ADDAHook))
        log_files = common_log_files()
        log_files.update(
            {
                "optimizers_T_Adam": {"lr"},
                "optimizers_D_Adam": {"lr"},
                "engine_output_d_loss": {
                    "total",
                    "d_src_domain_loss",
                    "d_target_domain_loss",
                },
                "engine_output_g_loss": {"total", "g_target_domain_loss"},
                "hook_e8e3833076189a9b4fc7353b2718c544cb88b47118cf526be3904208e3267da2_SufficientAccuracy": {
                    "accuracy",
                    "threshold",
                },
                "hook_ADDAHook_hook_ChainHook_hooks1_OptimizerHook_weighter_MeanWeighter": {
                    "scale"
                },
                "hook_ADDAHook_hook_ChainHook_hooks0_OptimizerHook_weighter_MeanWeighter": {
                    "scale"
                },
            }
        )
        run_adapter(self, TEST_FOLDER, adapter, log_files)

    def test_aligner(self):
        log_files = common_log_files()
        log_files.update(
            {
                "optimizers_G_Adam": {"lr"},
                "optimizers_C_Adam": {"lr"},
                "engine_output_total_loss": {
                    "total",
                    "c_loss",
                    "features_confusion_loss",
                    "logits_confusion_loss",
                },
                "hook_AlignerPlusCHook_hook_ChainHook_hooks0_OptimizerHook_weighter_MeanWeighter": {
                    "scale"
                },
            }
        )
        for loss_fn in [MMDLoss(), CORALLoss()]:
            models = get_gcd()
            del models["D"]
            adapter = Aligner(models=models, hook_kwargs={"loss_fn": loss_fn})
            self.assertTrue(isinstance(adapter.hook, AlignerPlusCHook))
            run_adapter(self, TEST_FOLDER, adapter, log_files)

    def test_cdan(self):
        models = get_gcd()
        misc = Misc({"feature_combiner": RandomizedDotProduct([512, 10], 512)})
        g_weighter = MeanWeighter(weights={"g_target_domain_loss": 0.5, "c_loss": 0.1})
        adapter = CDAN(models=models, misc=misc, hook_kwargs={"g_weighter": g_weighter})
        self.assertTrue(isinstance(adapter.hook, CDANHook))
        log_files = gan_log_files()
        log_files.update(
            {
                "hook_CDANHook_hook_ChainHook_hooks0_OptimizerHook_weighter_MeanWeighter": {
                    "scale"
                },
                "hook_CDANHook_hook_ChainHook_hooks1_OptimizerHook_weighter_MeanWeighter": {
                    "scale"
                },
                "hook_8c2a74151317b9315573314fafc0d8ad6e12f72a84433739f6f0762a4ca11ab0_weights": {
                    "g_target_domain_loss",
                    "c_loss",
                },
            }
        )
        run_adapter(self, TEST_FOLDER, adapter, log_files)

    def test_classifier(self):
        models = get_gcd()
        del models["D"]
        adapter = Classifier(models=models)
        self.assertTrue(isinstance(adapter.hook, ClassifierHook))
        log_files = common_log_files()
        log_files.update(
            {
                "optimizers_G_Adam": {"lr"},
                "optimizers_C_Adam": {"lr"},
                "engine_output_total_loss": {
                    "total",
                    "c_loss",
                },
                "hook_ClassifierHook_hook_ChainHook_hooks0_OptimizerHook_weighter_MeanWeighter": {
                    "scale"
                },
            }
        )
        run_adapter(self, TEST_FOLDER, adapter, log_files)

    def test_dann(self):
        adapter = DANN(models=get_gcd())
        self.assertTrue(isinstance(adapter.hook, DANNHook))
        log_files = common_log_files()
        log_files.update(
            {
                "optimizers_G_Adam": {"lr"},
                "optimizers_C_Adam": {"lr"},
                "optimizers_D_Adam": {"lr"},
                "engine_output_total_loss": {
                    "total",
                    "c_loss",
                    "src_domain_loss",
                    "target_domain_loss",
                },
                "hook_DANNHook_hook_ChainHook_hooks0_OptimizerHook_weighter_MeanWeighter": {
                    "scale"
                },
                "hook_5687ba61e823ab5ff4b1165a67973f4d25f9dc886d37ed6c6f17d900bcdba589_GradientReversal": {
                    "weight"
                },
            }
        )
        run_adapter(self, TEST_FOLDER, adapter, log_files)

    def test_domain_confusion(self):
        models = get_gcd()
        models["D"] = Discriminator(in_size=512, out_size=2)
        adapter = DomainConfusion(models=models)
        self.assertTrue(isinstance(adapter.hook, DomainConfusionHook))
        log_files = gan_log_files()
        log_files.update(
            {
                "hook_DomainConfusionHook_hook_ChainHook_hooks0_OptimizerHook_weighter_MeanWeighter": {
                    "scale"
                },
                "hook_DomainConfusionHook_hook_ChainHook_hooks1_OptimizerHook_weighter_MeanWeighter": {
                    "scale"
                },
            }
        )
        run_adapter(self, TEST_FOLDER, adapter, log_files)

    def test_finetuner(self):
        models = get_gcd()
        del models["D"]
        adapter = Finetuner(models=models)
        self.assertTrue(isinstance(adapter.hook, FinetunerHook))
        log_files = common_log_files()
        log_files.update(
            {
                "optimizers_C_Adam": {"lr"},
                "engine_output_total_loss": {
                    "total",
                    "c_loss",
                },
                "hook_FinetunerHook_hook_ChainHook_hooks0_OptimizerHook_weighter_MeanWeighter": {
                    "scale"
                },
            }
        )
        run_adapter(self, TEST_FOLDER, adapter, log_files)

    def test_gan(self):
        adapter = GAN(models=get_gcd())
        self.assertTrue(isinstance(adapter.hook, GANHook))
        log_files = gan_log_files()
        log_files.update(
            {
                "hook_GANHook_hook_ChainHook_hooks0_OptimizerHook_weighter_MeanWeighter": {
                    "scale"
                },
                "hook_GANHook_hook_ChainHook_hooks1_OptimizerHook_weighter_MeanWeighter": {
                    "scale"
                },
            }
        )
        run_adapter(self, TEST_FOLDER, adapter, log_files)

    def test_joint_aligner(self):
        models = get_gcd()
        del models["D"]
        aligner_hook = JointAlignerHook()
        adapter = Aligner(models=models, hook_kwargs={"aligner_hook": aligner_hook})
        self.assertTrue(isinstance(adapter.hook, AlignerPlusCHook))
        log_files = common_log_files()
        log_files.update(
            {
                "optimizers_G_Adam": {"lr"},
                "optimizers_C_Adam": {"lr"},
                "engine_output_total_loss": {
                    "total",
                    "c_loss",
                    "joint_confusion_loss",
                },
                "hook_AlignerPlusCHook_hook_ChainHook_hooks0_OptimizerHook_weighter_MeanWeighter": {
                    "scale"
                },
            }
        )
        run_adapter(self, TEST_FOLDER, adapter, log_files)

    def test_gvb(self):
        models = get_gcd()
        models["D"] = Discriminator(in_size=10)
        adapter = GVB(models=models)
        self.assertTrue(isinstance(adapter.hook, GVBHook))
        log_files = common_log_files()
        log_files.update(
            {
                "optimizers_G_Adam": {"lr"},
                "optimizers_C_Adam": {"lr"},
                "optimizers_D_Adam": {"lr"},
                "engine_output_total_loss": {
                    "total",
                    "c_loss",
                    "src_domain_loss",
                    "target_domain_loss",
                    "g_src_bridge_loss",
                    "g_target_bridge_loss",
                    "d_src_bridge_loss",
                    "d_target_bridge_loss",
                },
                "hook_GVBHook_hook_ChainHook_hooks0_OptimizerHook_weighter_MeanWeighter": {
                    "scale"
                },
                "hook_dc38aabd8a56ccc0cbd6a1a5a3be207eb683ec156091ecb1a9523d66e3c1914b_GradientReversal": {
                    "weight"
                },
            }
        )
        run_adapter(self, TEST_FOLDER, adapter, log_files)

    def test_mcd(self):
        models = get_gcd()
        models["C"] = MultipleModels(
            models["C"], c_f.reinit(copy.deepcopy(models["C"]))
        )
        del models["D"]
        adapter = MCD(models=models)
        self.assertTrue(isinstance(adapter.hook, MCDHook))
        log_files = common_log_files()
        log_files.update(
            {
                "optimizers_G_Adam": {"lr"},
                "optimizers_C_Adam": {"lr"},
                "engine_output_x_loss": {
                    "total",
                    "c_loss0",
                    "c_loss1",
                },
                "engine_output_y_loss": {
                    "total",
                    "c_loss0",
                    "c_loss1",
                    "discrepancy_loss",
                },
                "engine_output_z_loss": {"total", "discrepancy_loss"},
                "hook_c518bcf00aa142722b6625e934b61df8e3571627c09b6f91337b988b54c08aff_MeanWeighter": {
                    "scale"
                },
                "hook_5a4884ddfc716ba163f88c757cfd2926ad520f7c3c05786c17e322e0a6b384c4_MeanWeighter": {
                    "scale"
                },
                "hook_c9a8cb2a11004ce000995e0ee6b50351db47d82a447246571f7b99750d61dcfb_MeanWeighter": {
                    "scale"
                },
            }
        )
        run_adapter(self, TEST_FOLDER, adapter, log_files)

    def test_rtn(self):
        models = get_gcd()
        misc = Misc({"feature_combiner": RandomizedDotProduct([512, 10], 512)})
        models["residual_model"] = PlusResidual(torch.nn.Linear(10, 10))
        del models["D"]
        adapter = RTN(models=models, misc=misc)
        self.assertTrue(isinstance(adapter.hook, RTNHook))
        log_files = common_log_files()
        log_files.update(
            {
                "optimizers_G_Adam": {"lr"},
                "optimizers_C_Adam": {"lr"},
                "optimizers_residual_model_Adam": {"lr"},
                "engine_output_total_loss": {
                    "total",
                    "c_loss",
                    "entropy_loss",
                    "features_confusion_loss",
                },
                "hook_RTNHook_hook_ChainHook_hooks0_OptimizerHook_weighter_MeanWeighter": {
                    "scale"
                },
            }
        )
        run_adapter(self, TEST_FOLDER, adapter, log_files)

    def test_symnets(self):
        models = get_gcd()
        models["C"] = MultipleModels(
            models["C"], c_f.reinit(copy.deepcopy(models["C"]))
        )
        del models["D"]
        adapter = SymNets(models=models)
        self.assertTrue(isinstance(adapter.hook, SymNetsHook))
        log_files = common_log_files()
        log_files.update(
            {
                "optimizers_G_Adam": {"lr"},
                "optimizers_C_Adam": {"lr"},
                "engine_output_c_loss": {
                    "c_loss0",
                    "c_loss1",
                    "c_symnets_src_domain_loss_0",
                    "c_symnets_target_domain_loss_1",
                    "total",
                },
                "engine_output_g_loss": {
                    "symnets_category_loss",
                    "g_symnets_target_domain_loss_0",
                    "g_symnets_target_domain_loss_1",
                    "symnets_entropy_loss",
                    "total",
                },
                "hook_SymNetsHook_hook_ChainHook_hooks1_OptimizerHook_weighter_MeanWeighter": {
                    "scale"
                },
                "hook_SymNetsHook_hook_ChainHook_hooks2_OptimizerHook_weighter_MeanWeighter": {
                    "scale"
                },
            }
        )
        run_adapter(self, TEST_FOLDER, adapter, log_files)

    def test_vada(self):
        adapter = VADA(models=get_gcd())
        self.assertTrue(isinstance(adapter.hook, VADAHook))
        log_files = gan_log_files()
        log_files.update(
            {
                "hook_VADAHook_hook_ChainHook_hooks0_OptimizerHook_weighter_MeanWeighter": {
                    "scale"
                },
                "hook_VADAHook_hook_ChainHook_hooks1_OptimizerHook_weighter_MeanWeighter": {
                    "scale"
                },
                "hook_08bfa851645abcfce017c8e49f585b35d61121796abef262b9b4ffacde9fa96e_VATLoss": {
                    "num_power_iterations",
                    "xi",
                    "epsilon",
                },
            }
        )
        log_files["engine_output_g_loss"].update(
            {"src_vat_loss", "target_vat_loss", "entropy_loss"}
        )
        run_adapter(self, TEST_FOLDER, adapter, log_files)
