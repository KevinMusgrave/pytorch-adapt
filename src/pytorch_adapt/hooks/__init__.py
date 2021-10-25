from .adabn import AdaBNHook
from .adda import ADDAHook
from .aligners import (
    AlignerHook,
    AlignerPlusCHook,
    FeaturesLogitsAlignerHook,
    JointAlignerHook,
    ManyAlignerHook,
)
from .atdoc import ATDOCHook
from .cdan import CDANDomainHookD, CDANDomainHookG, CDANEHook, CDANHook, GANEHook
from .classification import ClassifierHook, CLossHook, FinetunerHook, SoftmaxHook
from .conditions import StrongDHook
from .dann import (
    CDANNEHook,
    DANNEHook,
    DANNHook,
    DANNLogitsHook,
    DANNSoftmaxLogitsHook,
    GradientReversalHook,
    GradientReversalLocallyHook,
    SoftmaxGradientReversalHook,
    SoftmaxGradientReversalLocallyHook,
)
from .domain import DomainLossHook, FeaturesForDomainLossHook
from .domain_confusion import DomainConfusionHook
from .features import (
    CombinedFeaturesHook,
    DLogitsHook,
    FeaturesAndLogitsHook,
    FeaturesHook,
    FeaturesWithGradAndDetachedHook,
    FrozenModelHook,
    LogitsHook,
)
from .gan import GANHook
from .gvb import GVBEHook, GVBGANHook, GVBHook
from .itl import ISTLossHook
from .losses import (
    AFNHook,
    BaseLossHook,
    BNMHook,
    BSPHook,
    MCCHook,
    TargetDiversityHook,
    TargetEntropyHook,
)
from .mcd import MCDHook, MCDLossHook, MultipleCLossHook
from .reducers import EntropyReducer, MeanReducer, MultipleReducers
from .rtn import ResidualHook, RTNAlignerHook, RTNHook, RTNLogitsHook
from .symnets import (
    SymNetsCategoryLossHook,
    SymNetsCHook,
    SymNetsDomainLossHook,
    SymNetsEntropyHook,
    SymNetsGHook,
    SymNetsHook,
)
from .utils import (
    ApplyToListHook,
    AssertHook,
    ChainHook,
    EmptyHook,
    FalseHook,
    ParallelHook,
    RepeatHook,
    TrueHook,
)
from .vada import VADAHook, VATHook, VATPlusEntropyHook
from .validate import validate_hook
