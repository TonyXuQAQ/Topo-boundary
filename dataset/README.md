# Dataset preparation
```
# File structure
├── dataset
|   ├── data_split.json
|   ├── config_dir.yml
|   ├── get_data.bash
|   ├── get_checkpoints.bash
│   ├── cropped_tiff
│   ├── labels
|   ├── pretrain_checkpoints
│   └── scripts
```

**Detailed data definition and structure can be found in the [supplementray document](./topoboundary_supplementary.pdf).**

## Dataset Download
Download and unzip```cropped_tiff``` and ```labels``. All the data are stored in Google Drive. To download all the data, run
```
# install gdown
pip install gdown
# optional 
chmod +x ./get_data.bash
# download and unzip the data from Google Drive
./get_data.bash
```
In case the script fails, you can download and unzip the data manually [aerial image data](https://drive.google.com/file/d/1xasG1LEeBuB-MmdiMGaX1jgZN8ThW3Jd/view?usp=sharing), [labels](https://drive.google.com/file/d/1XoAjhkwbO6IaYURikrKteA17NVlLoyf8/view?usp=sharing).

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
python ./scripts/generate_inv_dir_maps.py
```
to generate the inverse distance map and the direction map. Note it may take a lot of disk usage.

## Download saved pretrain checkpoints
We provide the saved checkpoints of our implementations. Run 
```
./get_checkpoints.bash
```
to download them, and the checkpoints are saved in ```./pretrain_checkpoints```. Checkpoints are copied to corresponding baseline folders automatically. You can also download it manually [pretrain checkpoints](https://drive.google.com/file/d/1OT8Vrj1QiT7zFgu-D4ZcN4dRR62U31CL/view?usp=sharing).