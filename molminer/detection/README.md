# Chemical Data Extraction
This repo is for tracking code and materials related to the 2024 Compoutational Drug Design (CDD) summer intern project by Abhisek Dey.

## Stage - 1 (Molecule Localization/Detection)

This stage was trained using the [Ultralytics](https://docs.ultralytics.com/) YoloV8 pipeline code using [DPRL Lab's](https://www.cs.rit.edu/~dprl/) collated version of the [CLEF-IP 2012](https://www.ifs.tuwien.ac.at/~clef-ip/2012/chem.shtml) dataset for molecular structure recognition task.

### Requirements

```shell
conda create -n yolov8 python=3.10
conda activate yolov8
pip install -r requirements.txt
```

### Patent Data and Model Weights (S3 Storage)
Best Model Weights are available at `s3://2025-molecule-miner/weights/detection.pt`.
Relevant Patents with generated structures is available at `s3://2025-molecule-miner/Test_Patents`

**Note:** The processed images and the CSV detections are now also available at `s3://2025-molecule-miner/Test_Patents`. No need to run training or evaluation again

* The weights should be copied to the root of this repo from S3 (Run from Root of this repository):
    `aws s3 cp s3://2025-molecule-miner/Detection_Weights/yolov8n.pt yolov8n.pt`
* The PDFs and annotation files should be copied to the folder `datasets/data`:
    `aws s3 cp s3://2025-molecule-miner/Test_Patents datasets/data/Test_Patents --recursive`

If recrusive copy fails, try copying the files individually:
```shell
aws s3 cp s3://2025-molecule-miner/Test_Patents/WO2013086208A1.pdf datasets/data/Test_Patents/
aws s3 cp s3://2025-molecule-miner/Test_Patents/WO2017189823A2.pdf datasets/data/Test_Patents/
aws s3 cp s3://2025-molecule-miner/Test_Patents/WO2018234343A1.pdf datasets/data/Test_Patents/
aws s3 cp s3://2025-molecule-miner/Test_Patents/WO2021004535A1.pdf datasets/data/Test_Patents/
````

### Training

**Note:** Training is done using DPRL's refined version of the CLEF-IP 2012 dataset. To run training you MUST download and extract the dataset first from Amazon S3.

Run from root of this repo:
```bash
aws s3 cp s3://2025-molecule-miner/yoloclef_data.zip datasets/data/yolo_clef_data.zip
unzip yolo_clef_data.zip
```

To run training (100 Epochs):
```python
conda activate yolov8
python train.py
```

The new weights (best epoch and last epoch) should be saved in the `runs/detect/train` directory.

**Note for GPU training on EC2 instances:** By default the code will look for an available gpu and will occupy the first gpu (GPU:0) on the instance by default. Parallel training is not yet supported. If there are no GPU's available, it will train on the CPU (much slower). 

### Evaluation (On DYRK1A and MK2 Patents)

#### Generating Images from PDFs and then generating the CSVs for Detected Regions (Per PDF)

`python generate_detections.py \<Root Folder containing the Patent Classes\> \<Directory to store the images and the detections\>`

**Example Command (Run from Root of This Repo)**

This assumes that `Test_Patents` is the root directory of the Patent files downloaded from S3.
```
python local_visualize.py datasets/data/Test_Patents/ datasets/data/Test_Patents/processed
```

This should generate a folder structure inside `Test_Patents` called `processed` containing all the PDFs as Images and the corresponding detection CSVs at `Test_Patents/processed/detections`

### Visualizing the Patent Documents Overlaid with the Detected Molecule Regions

This assumes that PDF folders containing the page images and the CSV detections for each PDF is already available.

Example Command to Generate Overlay Boxes:

```shell
python visualize_detections.py --in_dir datasets/data/Test_Patents/processed --out_dir datasets/data/Test_Patents/processed/overlaid
```

where:
* `in_dir`: Input Directory containing processed PDFs and Detected CSV files
* `out_dir`: Output Directory to save the overlaid boxes

Successfully running this command would generate an `overlaid` directory containing all the PDFs buy folders with each folder containing all the page images with **any** detected regions.


## Collation of Detected Boxes and SMILES Info in Spreadsheets

Till now, only the patents `W02017189823A2` and `WO2013086208A1` are supported. A total of 1837 have been successfully collated from both of them. They can be found at `s3://2025-molecule-miner/MoleculeBank/`.

If you want to run the collation process yourself, make sure that the detection csvs, the excel spreadsheets and the patent images are already available. The paths to them are hardcoded in the script for now. In future, all the paths and parameters for collation will be available as individual YAML files (TODO: Feature Addition).

To run collation, run from root repo directory:

`python collate_tables.py`

This should create all the collated data (Molecule Images and associated CID-SMILES csv) in the directory `MoleculeBank`. It should be same in structure as the S3 storage directory.

## Authors, Maintainers and Acknowledgements

* Abhisek Dey (Insitro, Research Intern - Cheminformatics 2024) - Author and Maintainer
* Nate Stanley (Insitro, CDD, Director) - Mentor and Manager
* Matt Langsenkamp (DPRL, RIT, Research Programmer) - Refined the current version of DPRL's archive of the Molecular Structure Recognition Dataset



