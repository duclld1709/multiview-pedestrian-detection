"""Local BEV inference demo with preprocessed video caches."""

import os

# Inference, decoding and rectification are threaded separately.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
