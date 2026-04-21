## Object Instance Segmentation

- Preprocessed object instance segmentation data download:

```
./download_benchmark_dataset.sh -o <output_dir>
```

- Run dataset process from Multiscan Dataset:

```
python gen_instsegm_dataset.py input_path=/path/to/multiscan_dataset output_path=<object_instance_segmentation output_dir>
```

## Structure

dataset/
└── multiscan.glb # Multiscan dataset(not in Github)

CS479-Assignment-3DPC-Segmentation-Challenge/
├── assets/
│   ├── sample.glb         # Object mesh data
│   └── test_0000.npy      # Test data example
│
├── dataset.py             # Code for loading test dataset (provided, DO NOT MODIFY)
├── evaluate.py            # Evaluation script (provided, DO NOT MODIFY)
├── model.py               # Model definition (students SHOULD modify)
└── visualize.py           # Visualization script (provided, CAN modify)