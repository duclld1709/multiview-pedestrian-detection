from models.segnet import Segnet
from models.mvdet import MVDet
from models.liftnet import Liftnet
# BEVFormer needs mmcv CUDA ops; import lazily so the other models run without it
try:
    from models.bevformernet import Bevformernet
    BEVFORMER_IMPORT_ERROR = None
except ImportError as e:
    Bevformernet = None
    BEVFORMER_IMPORT_ERROR = e
