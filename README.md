# README

## Getting Started

This guide provides instructions on how to run the training and testing pipelines for the project.

### 1. Training

To start the training process, run the following script:

```bash
./united_var_plus_train_all.sh
```

**Important Configuration Notes:**
- **Model Selection**: Make sure to set the `model` parameter to `'SeriesTank'`.
- **Dataset Path**: Set the `camels_root` parameter to the directory where you have extracted the dataset.
- **GPU Selection**: Adjust the GPU settings in the script according to your machine's available hardware.

### 2. Output Directory

After launching the training job, a new folder named with your specified `model_id` will be automatically created under the `runs_paper/` directory. All training logs and checkpoints will be saved in this folder.

### 3. Testing

Once training is complete, run the testing script:

```bash
./united_var_plus_test_all.sh
```

**Important Configuration Notes for Testing:**
- **Run Directory**: Update the `run_dir` parameter in the script to point to the path of your `model_id` folder (e.g., `runs_paper/model_id`).
- **Dataset Path**: Ensure that `camels_root` is correctly set to your dataset directory, just as in the training step.

### 4. Test Results

After testing completes, the results will be generated and saved in the `test/` subdirectory within your `model_id` folder (e.g., `runs_paper/model_id/test/`).
