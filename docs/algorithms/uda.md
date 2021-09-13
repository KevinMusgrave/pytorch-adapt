# Unsupervised Domain Adaptation

### DANN
[Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818)

- [adapters.DANN][pytorch_adapt.adapters.DANN]
- [hooks.DANNHook][pytorch_adapt.hooks.DANNHook]
- [layers.GradientReversal][pytorch_adapt.layers.GradientReversal]

### MMD
[Learning Transferable Features with Deep Adaptation Networks](https://arxiv.org/abs/1502.02791)

- [adapters.Aligner][pytorch_adapt.adapters.Aligner]
- [hooks.AlignerPlusCHook][pytorch_adapt.hooks.AlignerPlusCHook]
- [layers.MMDLoss][pytorch_adapt.layers.MMDLoss]


### Domain Confusion
[Simultaneous Deep Transfer Across Domains and Tasks](https://arxiv.org/abs/1510.02192)

- [adapters.DomainConfusion][pytorch_adapt.adapters.DomainConfusion]
- [hooks.DomainConfusionHook][pytorch_adapt.hooks.DomainConfusionHook]
- [layers.UniformDistributionLoss][pytorch_adapt.layers.UniformDistributionLoss]


### CORAL
[Deep CORAL: Correlation Alignment for Deep Domain Adaptation](https://arxiv.org/abs/1607.01719)

- [adapters.Aligner][pytorch_adapt.adapters.Aligner]
- [hooks.AlignerPlusCHook][pytorch_adapt.hooks.AlignerPlusCHook]
- [layers.CORALLoss][pytorch_adapt.layers.CORALLoss]


### RTN
[Unsupervised Domain Adaptation with Residual Transfer Networks](https://arxiv.org/abs/1602.04433)

- [adapters.RTN][pytorch_adapt.adapters.RTN]
- [hooks.RTNHook][pytorch_adapt.hooks.RTNHook]
- [layers.PlusResidual][pytorch_adapt.layers.PlusResidual]


### JMMD
[Deep Transfer Learning with Joint Adaptation Networks](https://arxiv.org/abs/1605.06636)

- [adapters.Aligner][pytorch_adapt.adapters.Aligner]
- [hooks.AlignerPlusCHook][pytorch_adapt.hooks.AlignerPlusCHook]
- [hooks.JointAlignerHook][pytorch_adapt.hooks.JointAlignerHook]
- [layers.MMDLoss][pytorch_adapt.layers.MMDLoss]


### ADDA
[Adversarial Discriminative Domain Adaptation](https://arxiv.org/abs/1702.05464)

- [adapters.ADDA][pytorch_adapt.adapters.ADDA]
- [hooks.ADDAHook][pytorch_adapt.hooks.ADDAHook]
- [hooks.StrongDHook][pytorch_adapt.hooks.StrongDHook]


### AdaBN
[Revisiting Batch Normalization For Practical Domain Adaptation](https://arxiv.org/abs/1603.04779)

Docs coming soon


### GAN

- [adapters.GAN][pytorch_adapt.adapters.GAN]
- [hooks.GANHook][pytorch_adapt.hooks.GANHook]

### VADA
[A DIRT-T Approach to Unsupervised Domain Adaptation](https://arxiv.org/abs/1802.08735)

- [adapters.VADA][pytorch_adapt.adapters.VADA]
- [hooks.VADAHook][pytorch_adapt.hooks.VADAHook]
- [hooks.VATHook][pytorch_adapt.hooks.VATHook]
- [layers.VATLoss][pytorch_adapt.layers.VATLoss]
- [layers.EntropyLoss][pytorch_adapt.layers.EntropyLoss]


### MCD
[Maximum Classifier Discrepancy for Unsupervised Domain Adaptation](https://arxiv.org/abs/1712.02560)

- [adapters.MCD][pytorch_adapt.adapters.MCD]
- [hooks.MCDHook][pytorch_adapt.hooks.MCDHook]
- [layers.MCDLoss][pytorch_adapt.layers.MCDLoss]


### CDAN
[Conditional Adversarial Domain Adaptation](https://arxiv.org/abs/1705.10667)

- [adapters.CDAN][pytorch_adapt.adapters.CDAN]
- [hooks.CDANHook][pytorch_adapt.hooks.CDANHook]
- [hooks.EntropyReducer][pytorch_adapt.hooks.EntropyReducer]
- [layers.EntropyWeights][pytorch_adapt.layers.EntropyWeights]
- [layers.RandomizedDotProduct][pytorch_adapt.layers.RandomizedDotProduct]


### BSP
[Transferability vs. Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation](http://proceedings.mlr.press/v97/chen19i.html)

- [layers.BatchSpectralLoss][pytorch_adapt.layers.BatchSpectralLoss]


### AFN
[Larger Norm More Transferable: An Adaptive Feature Norm Approach for Unsupervised Domain Adaptation](https://arxiv.org/abs/1811.07456)

- [layers.AdaptiveFeatureNorm][pytorch_adapt.layers.AdaptiveFeatureNorm]
- [layers.L2PreservedDropout][pytorch_adapt.layers.L2PreservedDropout]


### SWD
[Sliced Wasserstein Discrepancy for Unsupervised Domain Adaptation](https://arxiv.org/abs/1903.04064)

- [adapters.MCD][pytorch_adapt.adapters.MCD]
- [hooks.MCDHook][pytorch_adapt.hooks.MCDHook]
- [layers.SlicedWasserstein][pytorch_adapt.layers.SlicedWasserstein]

### SymNets
[Domain-Symmetric Networks for Adversarial Domain Adaptation](https://arxiv.org/abs/1904.04663)

- [adapters.SymNets][pytorch_adapt.adapters.SymNets]
- [hooks.SymNets][pytorch_adapt.hooks.SymNetsHook]
- [layers.ConcatSoftmax][pytorch_adapt.layers.ConcatSoftmax]


### GVB
[Gradually Vanishing Bridge for Adversarial Domain Adaptation](https://arxiv.org/abs/2003.13183)

- [adapters.GVB][pytorch_adapt.adapters.GVB]
- [hooks.GVBHook][pytorch_adapt.hooks.GVBHook]
- [layers.ModelWithBridge][pytorch_adapt.layers.ModelWithBridge]


### STAR
[Stochastic Classifiers for Unsupervised Domain Adaptation](https://xiatian-zhu.github.io/papers/LuEtAl_CVPR2020.pdf)

- [adapters.MCD][pytorch_adapt.adapters.MCD]
- [hooks.MCDHook][pytorch_adapt.hooks.MCDHook]
- [layers.StochasticLinear][pytorch_adapt.layers.StochasticLinear]


### BNM
[Towards Discriminability and Diversity: Batch Nuclear-norm Maximization under Label Insufficient Situations](https://arxiv.org/abs/2003.12237)

- [layers.BNMLoss][pytorch_adapt.layers.BNMLoss]


### MCC
[Minimum Class Confusion for Versatile Domain Adaptation](https://arxiv.org/abs/1912.03699)

- [layers.MCCLoss][pytorch_adapt.layers.MCCLoss]


### ATDOC
[Domain Adaptation with Auxiliary Target Domain-Oriented Classifier](https://arxiv.org/abs/2007.04171)

- [hooks.ATDOCHook][pytorch_adapt.hooks.ATDOCHook]
- [layers.NeighborhoodAggregation][pytorch_adapt.layers.NeighborhoodAggregation]
- [layers.ConfidenceWeights][pytorch_adapt.layers.ConfidenceWeights]
