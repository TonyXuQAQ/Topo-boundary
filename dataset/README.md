# Dataset preparation
```
# File structure
├── dataset
|   ├── data_split.json
|   ├── config_dir.yml
|   ├── get_data_new.bash
|   ├── get_label.bash
|   ├── get_checkpoints.bash
│   ├── cropped_tiff
│   ├── labels
|   ├── pretrain_checkpoints
│   └── scripts
```

**Detailed data definition and structure can be found in the [supplementray document](./topoboundary_supplementary.pdf).**

## ~~Dataset Download (Duplicated! Data download method updated!! See the next section!!)~~
~~Download and unzip ```cropped_tiff``` and ```labels``. All the data are stored in Google Drive. To download all the data, run~~
<!-- ```
# install gdown
pip install gdown
# optional 
chmod +x ./get_data.bash
# download and unzip the data from Google Drive
./get_data.bash
``` -->
~~In case the script fails, you can download and unzip the data manually [aerial image data](https://drive.google.com/file/d/1xasG1LEeBuB-MmdiMGaX1jgZN8ThW3Jd/view?usp=sharing), [labels](https://drive.google.com/file/d/1XoAjhkwbO6IaYURikrKteA17NVlLoyf8/view?usp=sharing).~~

## New Dataset Download Method 
Latest update : Sep/17/2021

Since Google drive has restrictions on download bandwidth, ```cropped_tiff``` is blocked for free download frequently. This issue may be caused by the huge size of ```cropped_tiff```. Currently, a **new** data preparation script is provided. Run 
```
# new script to prepare tiff images
./get_data_new.bash
# old script to prepare labels
./get_data.bash
```
In this script, we directly download the raw aerial image data from NYC database and conduct a set of processings to obtain tiff images that we need. It may takes some time for download and processing.
<!-- The script should be able to work properly, but it has not been fully verified at this stage. Report bugs in issue if necessary. Our team will try our best to fix them. More test will be conducted in the future. -->

**Note**: Other data (e.g., labels and checkpoints) should be able to be downloaded from Google drive properly.

**Note**: ```PILLOW``` requires ```Openjpeg``` library. If you cannot run the script properly, install it with ```pip``` and reinstall ```PILLOW```.
```
pip install pylibjpeg-openjpeg
# skip this step if pillow is not installed
pip uninstall pillow
pip install pillow
```
## Data label download
Download and unzip ```labels```. All the data are stored in Google Drive. To download all the data, run
```
# install gdown
pip install gdown
# optional 
chmod +x ./get_data.bash
# download and unzip the label from Google Drive
./get_label.bash
```
**In case the script fails, you can download and unzip the data manually [labels](https://drive.google.com/file/d/19Ti2Y-tE2Pw027Vsk2egvMJ0bkfM1ANQ/view?usp=sharing).**

## Dataset splitting
 ```data_split.json``` defines how the dataset is split into pretrain/train/valid/test. They are randomly split. It is recommended to use our provided data splitting file.
 
You can generate a new data split file by yourself by running
```
python ./scripts/split_data.py
```


## Set your directories
```config_dir.yml``` records the file directories. You should set the directory based on your own condition.

## Generate the inverse distance map and direction map
Since the inverse distance map and direction map are really huge so that we only provide the CUDA code to generate it without uploading it online. **Note** that these two labels are only for ConvBoundary baseline. 

If you are interested in this baseline model, run 
```
mkdir ./labels/inverse_distance_map
mkdir ./labels/direction_map
python ./scripts/generate_inv_dir_maps.py
```
to generate the inverse distance map and the direction map. Note it may take a lot of disk usage.

## Download saved pretrain checkpoints
We provide the saved checkpoints of our implementations. Run 
```
./get_checkpoints.bash
```
to download them, and the checkpoints are saved in ```./pretrain_checkpoints```. Checkpoints are copied to corresponding baseline folders automatically. **If the script cannot run, you can also download it manually [pretrain checkpoints](https://drive.google.com/file/d/1ijgnesWfvx5SfcuD68T5s8Lbr4ZclZ0R/view?usp=sharing).**
