In the docker/apptainer image, torch is version 1.21 and vmap need to be imported by `import vmap from functorch`.
`compare.py` and `models/_num_embedding.py` and `utils/io_utils_flat.py` are deprecated.
import catboost,pyarrow and geopy