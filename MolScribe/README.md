# MolscribeV2: Enhanced Positional Encodings for Feature-Sensitive Parsing

This is the implementation of enhancing molecular structure parsing using feature sensitive pre-training and positional encodings in a SWIN-Transformer based implementation. This code-base has been adapted from the **original Molscribe** [paper](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01480) ([GitHub repo](https://github.com/thomas0809/MolScribe)).

![Main-Pipeline](assets/main_pipeline.png)


## Getting Required libraries for training
MolScribe requires the Indigo Library for generating training images from SMILES data at training time. To do so, the libs can be safely downloaded from the S3 storage at 
`s3://2025-molecule-miner/MolScribe/lib/`. These should be inside the `MolScribe/molscribe/indigo/` directory. 

Assuming you are inside the indigo directory, run:
```shell
mkdir lib
aws s3 cp s3://2025-molecule-miner/MolScribe/lib/ lib/ --recursive --no-sign-request
```

## Train MolScribe (Ubuntu/Linux)
All requisite training data is available at `s3://2025-molecule-miner/MolScribe/data/`. A lot of the data was originally compiled by the original Molscribe authors. To download the training data, from the directory `MolScribe`, run:

### Downloading the training data
Training data for MolScribeV2 was compiled from PubChem, Zinc250k, ChemBL and MOSES datasets. There are about ~5 million training unique examples from these sets. To download that set:

```shell
mkdir data
aws s3 cp s3://2025-molecule-miner/MolScribe/data/ data/ --recursive --no-sign-request
```

### Downloading MolScribeV1 pre-trained weights 
Currently, ONLY the pre-trained decoder weights are being used to guide the new encoder training. This has been shown to improve training stability leading to better convergence.

You can download the appropriate weight file from this [link](https://www.dropbox.com/sh/91u508kf48cotv4/AACQden2waMXIqLwYSi8zO37a?dl=0&e=1&preview=swin_base_char_aux_1m680k.pth) (unfortunately wget does not work in this DropBox link anymore). Place it in the `MolScribe` directory of the repo.

### Starting training (Torchrun Distributed Node Cluster Training)
We use a distributed cluster of nodes each containing 8 GPUs for training. We provide the configuration script (`scripts/train_uspto_joint_chartok_5m_full.sh`)for training using torchrun. Please feel free to change the settings based on your hardware configuration. Do note that this can also be trained on a single GPU environment (not recommended) by setting `NUM_NODES=1` and `NUM_GPUS_PER_NODE=1`.

**Change torchrun training script path and the `BASE_MOUNT` environment variable  in the config to the correct path before training** 

To start training, from the directory `chem_data_extraction/MolScribe` run:
```shell
chmod +x scripts/train_uspto_joint_chartok_5m_full.sh
./scripts/train_uspto_joint_chartok_5m_full.sh
```
**IMPORTANT:** The number of GPUs are set to **4** and the batch size is set to **64** in the training script by default. Feel free to change the `BATCH_SIZE` and `NUM_GPUS_PER_NODE` arguments in the script to suit your hardware capabilities.

During training, we use a modified code of [Indigo](https://github.com/epam/Indigo) (included in `molscribe/indigo/`).

### Visualizing training progress on TensorBoard (Remote Workstation Instructions - EC2 etc.)
Now you can also visualize the training process including losses and gradients per mini-batch for your training run. To visualize in real time, on a separate shell window, login to your workstation by forwarding your system port:

`ssh -L 16006:localhost:6006 ec2-user@<ec2-ip-address>`

**Note:** This assumes that your public key is already registered with your EC2 Instance

Then go the MolScribe directory of this repo and run the following commands:
```shell
conda activate molscribe
tensorboard --logdir output/uspto/swin_base_char_aux_1m680k
```

Go to your local browser and go the address: `http://127.0.0.1:16006/`. You should now see TensorBoard server runnning on port `6006` on the remote server being mapped to port `16006` on your local machine.

## Important Training Parameters (ENV Vars and training flags)
For training there are a number of flags you can set (or unset) to make sure for your desired training routine. This list below is not exhaustive. Please refer to the arguments in `train.py` for a complete list of supported flags. You can change any environment variable/flag in the training script `scripts/train_uspto_joint_chartok_5m_full.sh`.

*  `--custom_enc`: If specified, uses the custom adaptation of SWIN-B Transformer instead of the default HuggingFace imnplementation. (Note: All extra additions to base MolScribe only work on this mode)
* `--train_v1`: 1 -> Uses the 1M pubchem data only for training 0 -> Uses the 5M custom collated dataset
* `--load_path`: If specified, will start training from the pre-trained checkpoint (Both Encoder and Decoder)
* `--mask_pos_emb`: 1 -> Custom Pixel Masking Enabled 0 -> Not enabled
* `--perturb`: If specified, uses Cropping and Edge adding augmentations
* `do_mops`: If specified, uses document degradation strategies (Dilation and Erosion)

## Evaluation Script (New: V2)

These are the steps to evaluate Molscribe V1 or V2 on **Molecule Bank data**. (Currently only V1 weights are available)

**Step - 1 (Generating Canonical SMILES Match):** This assumes that the detection data is already available. If not, please go to the `detection` sub-folder of this repo to generate detections. 

Download the V1 model checkpoint from [Dropbox](https://www.dropbox.com/sh/91u508kf48cotv4/AACQden2waMXIqLwYSi8zO37a?dl=0) namely the checkpoint `swin_base_char_aux_1m.pth` and move it to the `Molscribe` directory (not the root repo directory). 

Then run:

```shell
python evaluate_testset.py --csv_dir <path to the csv directory containing detection boxes> --pdf_mol_imgs <Directory containing the molecule images per PDF> --ckpt_path <path to the saved checkpoint model>
```
The defaults are:
* `csv_dir`: /home/ec2-user/my_work/chem_data_extraction/detection/MoleculeBank/csvs
* `pdf_mol_imgs`: /home/ec2-user/my_work/chem_data_extraction/detection/MoleculeBank
* `ckpt_path`: swin_base_char_aux_1m.pth

The following script will generate the Canonnical SMILES match in the terminal as well a `PredVSGT.txt` file. This file will be used in Step-2.

The result contains three scores on the terminal:
- canon_smiles: the main metric, exact matching accuracy.
- graph: graph exact matching accuracy, ignoring tetrahedral chirality.
- chiral: exact matching accuracy on chiral molecules.

**Step - 2 (Generating Tanimoto Similarity and Computing Graph-Edit Distance):** To evaluate the Tanimoto Distance and Graph-Edit Distance, using the `PredVSGT.txt` file, run from the `MolScribe` directory:

`python custom_utils/custom_metrics.py` *Note:* This assumes that the txt file is in the same directory.

## Author, Maintainer and Acknowledgements
* Abhisek Dey (Insitro, Research Intern - Cheminformatics 2024) - Author and Maintainer
* Nate Stanley (Insitro, CDD, Director) - Mentor and Manager
* Srinivasan Sivanandan (Insitro, Senior ML Scientist) - Advisory