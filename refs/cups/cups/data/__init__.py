from .cityscapes import (
    CITYSCAPES_CLASSNAMES,
    CITYSCAPES_CLASSNAMES_7,
    CITYSCAPES_CLASSNAMES_19,
    CITYSCAPES_STUFF_CLASSES,
    CITYSCAPES_STUFF_CLASSES_7,
    CITYSCAPES_STUFF_CLASSES_19,
    CITYSCAPES_THING_CLASSES,
    CITYSCAPES_THING_CLASSES_7,
    CITYSCAPES_THING_CLASSES_19,
    CITYSCAPES_VOID_CLASS,
    CityscapesLabelEfficient,
    CityscapesPanoptic,
    CityscapesPanopticValidation,
    CityscapesSelfTraining,
    CityscapesStereoVideo,
    CityscapesStereoVideoPanoptic,
    collate_function_detectron2_train,
    collate_function_validation,
)
from .kitti import (
    KITTI_INSTANCE_STUFF_CLASSES,
    KITTI_INSTANCE_THING_CLASSES,
    KITTIInstanceSegmentation,
    KITTIPanopticValidation,
    KITTIRaw,
    KITTISelfTraining,
    KITTIStereoVideo,
)
from .repeat_factor_sampler import RepeatFactorTrainingSampler
from .utils import StepDataset

try:
    from .bdd import BDD10kPanopticValidation
    from .mots import MOTS, MOTS_STUFF_CLASSES, MOTS_THING_CLASSES
    from .muses import MUSESPanopticValidation
    from .pseudo_label_dataset import PseudoLabelDataset
    from .waymo import (
        WAYMO_7_MISSING_CS_CLASSES,
        WAYMO_19_MISSING_CS_CLASSES,
        WAYMO_27_MISSING_CS_CLASSES,
        WaymoPanopticValidation,
    )
except ModuleNotFoundError as exc:
    if exc.name != "detectron2":
        raise
    # The official pseudo-label generator only needs CityscapesStereoVideo and
    # KITTIRaw. Training/validation datasets still load in full Detectron2 envs.
    pass
