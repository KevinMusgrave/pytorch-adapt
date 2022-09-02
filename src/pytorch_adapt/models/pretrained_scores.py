import numpy as np


def domain_len_assertion(domain_list):
    if len(domain_list) > 1:
        raise ValueError("only 1 domain currently supported")
    if len(domain_list) == 0:
        return None
    return domain_list[0]


def pretrained_src_accuracy(dataset, src_domains, split, average):
    src_domain = domain_len_assertion(src_domains)

    mnist = {
        "mnist": {
            "train": {"micro": 0.9994333386421204, "macro": 0.9994416236877441},
            "val": {"micro": 0.9951000213623047, "macro": 0.9949950575828552},
        }
    }

    office31 = {
        "amazon": {
            "train": {"micro": 0.9875721335411072, "macro": 0.986396312713623},
            "val": {"micro": 0.9042553305625916, "macro": 0.9006039500236511},
        },
        "dslr": {
            "train": {"micro": 0.9974874258041382, "macro": 0.9946237802505493},
            "val": {"micro": 0.9900000095367432, "macro": 0.9892473220825195},
        },
        "webcam": {
            "train": {"micro": 1.0, "macro": 1.0},
            "val": {"micro": 0.9811320900917053, "macro": 0.9784945249557495},
        },
    }

    officehome = {
        "art": {
            "train": {"micro": 0.9953632354736328, "macro": 0.99530029296875},
            "val": {"micro": 0.8148148059844971, "macro": 0.7875152826309204},
        },
        "clipart": {
            "train": {"micro": 0.955899178981781, "macro": 0.9570373892784119},
            "val": {"micro": 0.7537227869033813, "macro": 0.7572566270828247},
        },
        "product": {
            "train": {"micro": 0.9909884333610535, "macro": 0.99105304479599},
            "val": {"micro": 0.9268018007278442, "macro": 0.9206826686859131},
        },
        "real": {
            "train": {"micro": 0.9845049977302551, "macro": 0.980871319770813},
            "val": {"micro": 0.8841742873191833, "macro": 0.868537187576294},
        },
    }

    domainnet126 = {
        "clipart": {
            "train": {"micro": 0.9741344451904297, "macro": 0.9643224477767944},
            "val": {"micro": 0.8382785320281982, "macro": 0.815248966217041},
        },
        "painting": {
            "train": {"micro": 0.9724614024162292, "macro": 0.9370613098144531},
            "val": {"micro": 0.8157435059547424, "macro": 0.7473534345626831},
        },
        "real": {
            "train": {"micro": 0.9799062013626099, "macro": 0.9784836769104004},
            "val": {"micro": 0.8968163728713989, "macro": 0.8921362161636353},
        },
        "sketch": {
            "train": {"micro": 0.9720315337181091, "macro": 0.9493532180786133},
            "val": {"micro": 0.8155379295349121, "macro": 0.7761528491973877},
        },
    }

    x = {
        "mnist": mnist,
        "office31": office31,
        "officehome": officehome,
        "domainnet126": domainnet126,
    }

    return np.round(x[dataset][src_domain][split][average], 4)


def pretrained_target_accuracy(dataset, src_domains, target_domains, split, average):
    src_domain = domain_len_assertion(src_domains)
    target_domain = domain_len_assertion(target_domains)

    mnist = {
        "mnist": {
            "mnistm": {
                "train": {"micro": 0.576329231262207, "macro": 0.5797358751296997},
                "val": {"micro": 0.5739362239837646, "macro": 0.5770031213760376},
            }
        }
    }

    office31 = {
        "amazon": {
            "dslr": {
                "train": {"micro": 0.8266331553459167, "macro": 0.8512204885482788},
                "val": {"micro": 0.7799999713897705, "macro": 0.7833333015441895},
            },
            "webcam": {
                "train": {"micro": 0.803459107875824, "macro": 0.8174535036087036},
                "val": {"micro": 0.7735849022865295, "macro": 0.7740142941474915},
            },
        },
        "dslr": {
            "amazon": {
                "train": {"micro": 0.6959609389305115, "macro": 0.694110095500946},
                "val": {"micro": 0.695035457611084, "macro": 0.6928660273551941},
            },
            "webcam": {
                "train": {"micro": 0.9433962106704712, "macro": 0.942440390586853},
                "val": {"micro": 0.9245283007621765, "macro": 0.9129030704498291},
            },
        },
        "webcam": {
            "amazon": {
                "train": {"micro": 0.7150465846061707, "macro": 0.7132347822189331},
                "val": {"micro": 0.7251772880554199, "macro": 0.7302651405334473},
            },
            "dslr": {
                "train": {"micro": 0.9899497628211975, "macro": 0.9890364408493042},
                "val": {"micro": 0.9800000190734863, "macro": 0.9811827540397644},
            },
        },
    }

    officehome = {
        "art": {
            "clipart": {
                "train": {"micro": 0.41151201725006104, "macro": 0.41807034611701965},
                "val": {"micro": 0.4192439913749695, "macro": 0.4325830340385437},
            },
            "product": {
                "train": {"micro": 0.6860039234161377, "macro": 0.6699330806732178},
                "val": {"micro": 0.7038288116455078, "macro": 0.6905300617218018},
            },
            "real": {
                "train": {"micro": 0.7667145133018494, "macro": 0.7531952857971191},
                "val": {"micro": 0.7729358077049255, "macro": 0.7549898624420166},
            },
        },
        "clipart": {
            "art": {
                "train": {"micro": 0.6017516851425171, "macro": 0.5499463081359863},
                "val": {"micro": 0.6193415522575378, "macro": 0.5711568593978882},
            },
            "product": {
                "train": {"micro": 0.6764291524887085, "macro": 0.6609126925468445},
                "val": {"micro": 0.6869369149208069, "macro": 0.6790676712989807},
            },
            "real": {
                "train": {"micro": 0.7047345638275146, "macro": 0.6891355514526367},
                "val": {"micro": 0.6961008906364441, "macro": 0.6748995184898376},
            },
        },
        "product": {
            "art": {
                "train": {"micro": 0.6002060770988464, "macro": 0.5802421569824219},
                "val": {"micro": 0.604938268661499, "macro": 0.5949786305427551},
            },
            "clipart": {
                "train": {"micro": 0.428121417760849, "macro": 0.41485580801963806},
                "val": {"micro": 0.4249713718891144, "macro": 0.4174821376800537},
            },
            "real": {
                "train": {"micro": 0.7618364691734314, "macro": 0.750948429107666},
                "val": {"micro": 0.7855504751205444, "macro": 0.7742912173271179},
            },
        },
        "real": {
            "art": {
                "train": {"micro": 0.687789797782898, "macro": 0.6575398445129395},
                "val": {"micro": 0.7037037014961243, "macro": 0.6948687434196472},
            },
            "clipart": {
                "train": {"micro": 0.4470217525959015, "macro": 0.44673898816108704},
                "val": {"micro": 0.4387170672416687, "macro": 0.450015664100647},
            },
            "product": {
                "train": {"micro": 0.7907631397247314, "macro": 0.7783608436584473},
                "val": {"micro": 0.7860360145568848, "macro": 0.7748997211456299},
            },
        },
    }

    domainnet126 = {
        "clipart": {
            "painting": {
                "train": {"micro": 0.36399349570274353, "macro": 0.3765648603439331},
                "val": {"micro": 0.37343278527259827, "macro": 0.3998126983642578},
            },
            "real": {
                "train": {"micro": 0.44124647974967957, "macro": 0.4500534236431122},
                "val": {"micro": 0.4425809979438782, "macro": 0.4535055160522461},
            },
            "sketch": {
                "train": {"micro": 0.46875157952308655, "macro": 0.4666426479816437},
                "val": {"micro": 0.45596909523010254, "macro": 0.4594738483428955},
            },
        },
        "painting": {
            "clipart": {
                "train": {"micro": 0.5088223218917847, "macro": 0.4968717098236084},
                "val": {"micro": 0.5046778917312622, "macro": 0.4901778995990753},
            },
            "real": {
                "train": {"micro": 0.631933331489563, "macro": 0.619450569152832},
                "val": {"micro": 0.63978111743927, "macro": 0.6281707882881165},
            },
            "sketch": {
                "train": {"micro": 0.4545639455318451, "macro": 0.4474347233772278},
                "val": {"micro": 0.44803741574287415, "macro": 0.4399655759334564},
            },
        },
        "real": {
            "clipart": {
                "train": {"micro": 0.5694425702095032, "macro": 0.5809699892997742},
                "val": {"micro": 0.5680299401283264, "macro": 0.5786350965499878},
            },
            "painting": {
                "train": {"micro": 0.6136661171913147, "macro": 0.5830718278884888},
                "val": {"micro": 0.6183145642280579, "macro": 0.5833742618560791},
            },
            "sketch": {
                "train": {"micro": 0.4794813096523285, "macro": 0.4843434691429138},
                "val": {"micro": 0.47000202536582947, "macro": 0.47442907094955444},
            },
        },
        "sketch": {
            "clipart": {
                "train": {"micro": 0.5638283491134644, "macro": 0.5414069890975952},
                "val": {"micro": 0.5608125925064087, "macro": 0.5309245586395264},
            },
            "painting": {
                "train": {"micro": 0.4681163430213928, "macro": 0.41524365544319153},
                "val": {"micro": 0.4756387770175934, "macro": 0.43400508165359497},
            },
            "real": {
                "train": {"micro": 0.46748748421669006, "macro": 0.4649970531463623},
                "val": {"micro": 0.4728538990020752, "macro": 0.47053515911102295},
            },
        },
    }

    x = {
        "mnist": mnist,
        "office31": office31,
        "officehome": officehome,
        "domainnet126": domainnet126,
    }
    return np.round(x[dataset][src_domain][target_domain][split][average], 4)
