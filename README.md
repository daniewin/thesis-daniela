
# Code Repository of Master's Thesis "Analysing the Impact of Prompt Variability on Distilled Knowledge in Anomaly Segmentation" by Daniela Winter

This repository contains the code for the Master's Thesis "Analysing the Impact of Prompt Variability on Distilled Knowledge in Anomaly Segmentation". 
The code is structured in a modular way, with separate scripts for generating soft labels from the teacher model, training and testing the student model, as well as utility scripts for managing data, generating metrics and combination strategies for soft labels from multiple teachers.

## Dataset Examples
Exemplary images from ISEAD marine bird segmentation dataset:

<img src="images/example_1.jpg" alt="Dataset Example" width="100px"/>   <img src="images/example_2.jpg" alt="Dataset Example" width="100px"/>  <img src="images/example_3.jpg" alt="Dataset Example" width="100px"/>  <img src="images/example_4.jpg" alt="Dataset Example" width="100px"/>

Exemplary soft label masks created by teacher model SAA:

<img src="images/example_1_soft_label.png" alt="Dataset Example" width="100px"/>   <img src="images/example_2_soft_label.png" alt="Dataset Example" width="100px"/>  <img src="images/example_3_soft_label.png" alt="Dataset Example" width="100px"/>  <img src="images/example_4_soft_label.png" alt="Dataset Example" width="100px"/>


## Repository Structure

- `analysis/` - Contains the script for evaluating the CSV files that contain the BBox information using the metrics BBox-Precision, BBox-Recall, and BBox-F1-score.
- `images/` - Contains exemplary images of the ISEAD marine bird segmentation dataset and their soft labels generated by the teacher model SAA.
- `student/` - Contains the scripts for training and testing the student model.
  - **`train.py`** - Script used for training and evaluating the student model.
  - **`test.py`** - Script used for testing the student model.
  - `data_loader_student.py` - Data loader for the student model.

  
- `teacher/` - Contains the scripts for generating soft labels from teacher model.
  - `SegmentAnyAnomaly/` - Code for teacher model SAA from https://github.com/caoyunkang/Segment-Any-Anomaly
  - **`soft_label_generation.py`** - Script for generating soft labels from teacher model.
  - `data_loader_teacher.py` - Data loader for the teacher model.

- `util/` - Utility scripts used across different tasks.
  - `multi_teacher/`
    - `combination_strategies.py` - Script for combining soft labels from different teachers (includes averaging and majority voting).
    - `group_sampling.ipynb` - Jupyter notebook showing how multi-teacher groups were sampled.
  - `csv_from_soft_labels.py` - Script for converting soft labels into CSV format.
  - `data_classes.py` - Definitions of data structures.
  - `masks_to_bboxes.py` - Script to convert segmentation masks to bounding boxes.
  - `metrics.py` - Functions for calculating evaluation metrics.
  - `visualization.py` - Script for visualizing outputs with bounding boxes.


- `config.json` - Configuration file with student model parameters, paths, and other settings for training and testing the student model.
- `conda_student.yaml` - Conda environment file to set up the required dependencies to run the scripts for training and testing the student.
- `conda_teacher.yaml` - Conda environment file to set up the required dependencies to run the scripts for generating soft labels with the teacher model.



## Getting Started

### Setting Up the Environment

To set up the environment, use the provided `.yaml` file to create a new Conda environment with all the necessary dependencies.


1. Ensure Conda is installed on your system.
2. Create the environment from the `.yaml` file:

   ```bash
   conda env create -f conda_student.yml
   ```
   or
   ```bash
   conda env create -f conda_teacher.yml
   ```

3. Once the environment is created, activate it:

   ```bash
   conda activate env_student
   ```
   or
   ```bash
   conda activate env_teacher
   ```



### Generating Soft Labels (Teacher Model):
- Soft labels are generated using the teacher model through the `teacher/soft_label_generation.py` script.
- The teacher model setup is described at https://github.com/caoyunkang/Segment-Any-Anomaly
- Make sure to specifiy the prompts that should be used for the predictions using SAA. Default prompts are

    - anomaly prompt: "bird"
    - object prompt: "sea"
    - anomaly area threshold: 0.05
    - anomaly count threshold: 20
- Also set the correct folder of the input images and output folder in which the generated soft labels are saved. 


```bash
python teacher/soft_label_generation.py
```

### Combining the Soft Labels from Multiple Teachers

- The script `combination_strategies.py` located in the `util/multi_teacher/` directory is responsible for combining the output from multiple teacher models to create combined soft labels that are stored in the specified output folder and can be used for training the student model.

- Make sure that the soft labels are already generated and stored. These outputs are used as the input to the combination strategies.
- You can specify which combination strategy you would like to use (majority voting, averaging). This can be set in the script. 


```bash
python util/multi_teacher/combination_strategies.py
```



### Training the Student Model:
- The student model can be trained using the `student/train.py` script.
  
  ```bash
  python student/train.py
  ```

- This script requires the `config.json` to contain the model hyperparameters and the correct data paths to access the generated soft labels and to store the student model checkpoints.
- While commented out per default, Weights and Biases experiment tracking can be used to monitor the training process. For the validation the evaluation metrics are computed using `util/metrics.py`.


### Testing the Student Model:
- The student model can be tested using the `student/test.py` script.
  
  ```bash
  python student/test.py
  ```

- This script will load the trained student model, run it on the test dataset, and if the visualization flag is set to `True`, the predicted masks including a bounding box visualization are stored in the output folder specified in `config.json`.




