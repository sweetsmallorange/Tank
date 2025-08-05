## Getting Started

This guide provides instructions on how to run the training and testing pipelines for the project.

### Dataset Preparation

Before starting training, you need to download and set up the required datasets.

1. **Download the CAMELS dataset**:
   - Go to the official CAMELS website: [https://ral.ucar.edu/solutions/products/camels](https://ral.ucar.edu/solutions/products/camels)
   - Download the CAMELS dataset.
   - Extract the contents to a directory of your choice (e.g., `./data/camels/`).

2. **Download the PET (Potential Evapotranspiration) data**:
   - Download the `pet_harg.zip` file from Zenodo: [https://zenodo.org/records/7943626](https://zenodo.org/records/7943626)
   - Extract the `pet_harg.zip` file and place the resulting folder into the same directory as the CAMELS data.

3. **Set your `camels_root` path**:
   - The root directory (referred to as `camels_root` in the scripts) should now contain both the original CAMELS data and the `pet_hargreaves` (or similarly named) folder.

### 1. Training

To start the training process, run the following script:

```bash
./united_var_plus_train_all.sh
```

**Important Configuration Notes:**
- **Model Selection**: Make sure to set the `model` parameter to `'SeriesTank'`.
- **Dataset Path**: Set the `camels_root` parameter in the script to the directory you prepared in **Step 0** (e.g., `./data/camels/`).
- **GPU Selection**: Adjust the GPU settings in the script according to your machine's available hardware.

### 2. Output Directory

After launching the training job, a new folder named with your specified `model_id` will be automatically created under the `runs_paper/` directory. All training logs and checkpoints will be saved in this folder.

### 3. Testing

Once training is complete, run the testing script:

```bash
./united_var_plus_test_all.sh
```

**Important Configuration Notes for Testing:**
- **Run Directory**: Update the `run_dir` parameter in the script to point to your `model_id` folder (e.g., `runs_paper/model_id`).
- **Dataset Path**: Ensure that `camels_root` is correctly set to the same dataset directory used during training.

### 4. Test Results

After testing completes, the results will be generated and saved in the `test/` subdirectory within your `model_id` folder.

You can now analyze the performance of your model using these output files.


