# Efficient Localized Adaptation of Neural Weather Forecasting: A Case Study in the MENA Region 

[Paper(arXiv)](TBA)

Authors: [Muhammad Akhtar Munir](https://scholar.google.com.pk/citations?user=sT-epZAAAAAJ&hl=en), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en), [Salman Khan](https://salman-h-khan.github.io/)

This repository contains the PyTorch implementation of Efficient Localized Adaptation of Neural Weather Forecasting: A Case Study in the MENA Region

## Abstract
Accurate weather and climate modeling is critical for both scientific advancement and safeguarding communities against environmental risks. Traditional approaches rely heavily on Numerical Weather Prediction (NWP) models, which simulate energy and matter flow across Earth's systems. However, heavy computational requirements and low efficiency restrict the suitability of NWP, leading to a pressing need for enhanced modeling techniques. Neural network-based models have emerged as promising alternatives, leveraging data-driven approaches to forecast atmospheric variables. In this work, we focus on limited-area modeling and train our model specifically for localized region-level downstream tasks. As a case study, we consider the MENA region due to its unique climatic challenges, where accurate localized weather forecasting is crucial for managing water resources, agriculture and mitigating the impacts of extreme weather events. This targeted approach allows us to tailor the model's capabilities to the unique conditions of the region of interest. Our study aims to validate the effectiveness of integrating parameter-efficient fine-tuning (PEFT) methodologies, specifically Low-Rank Adaptation (LoRA) and its variants, to enhance forecast accuracy, as well as training speed, computational resource utilization, and memory efficiency in weather and climate modeling for specific regions.

![alt text](ClimaX_vis_mena.png)

Error/Bias in Predictions and Actual measurements for temperature\_2m (K). Dated, 11th April 2017, lead time 3 days

## Results

### Global vs Regional
Forecasting on MENA region for 72 hrs prediction. Resolution is $1.40652^\circ$. The global model performs worse whereas regional model, specific to the needs of the local region performs better.
Link to global and regional models can be found [link](https://drive.google.com/drive/folders/1PZyo3u-n9hk66ik2dAuaFSYd4fS_9t9G?usp=sharing)

| **Metric**                       | **Model**  | **geop@500** | **2m_temp** | **r_hum@850** | **s_hum@850** | **temp@850** | **10m_u_wind** | **10m_v_wind** |
|----------------------------------|------------|---------------|--------------|----------------|---------------|---------------|-----------------|-----------------|
| **ACC** (↑)                      | Global     | 0.292         | 0.230        | 0.255          | 0.282         | 0.246         | 0.287           | 0.238           |
|                                  | Regional   | 0.585         | 0.804        | 0.502          | 0.623         | 0.620         | 0.570           | 0.517           |
| **RMSE** (↓)                     | Global     | 674.295       | 3.349        | 23.308         | 0.003         | 3.561         | 3.733           | 4.162           |
|                                  | Regional   | 411.125       | 1.518        | 18.945         | 0.002         | 2.366         | 2.931           | 3.219           |

## Installation Guide


```
git clone https://github.com/microsoft/ClimaX
```

```
cd ClimaX
conda env create --file docker/environment.yml
conda activate climaX
```

```
# install so the project is in PYTHONPATH
pip install -e .
```

For complete Installation and usage instructions, follow the guidelines [here](https://github.com/microsoft/ClimaX/blob/main/docs/usage.md)


## Dataset and Training <[Instructions followed from this link](https://github.com/microsoft/ClimaX/blob/main/docs/usage.md)>

## Global Forecasting

### Data Preparation

First, download ERA5 data from [WeatherBench](https://dataserv.ub.tum.de/index.php/s/m1524895). The data directory should look like the following
```
5.625deg
   |-- 10m_u_component_of_wind
   |-- 10m_v_component_of_wind
   |-- 2m_temperature
   |-- constants.nc
   |-- geopotential
   |-- relative_humidity
   |-- specific_humidity
   |-- temperature
   |-- toa_incident_solar_radiation
   |-- total_precipitation
   |-- u_component_of_wind
   |-- v_component_of_wind
```

Then, preprocess the netcdf data into small numpy files and compute important statistics
```bash
python src/data_preprocessing/nc2np_equally_era5.py \
    --root_dir /mnt/data/5.625deg \
    --save_dir /mnt/data/5.625deg_npz \
    --start_train_year 1979 --start_val_year 2016 \
    --start_test_year 2017 --end_year 2019 --num_shards 8
```

The preprocessed data directory will look like the following
```
5.625deg_npz
   |-- train
   |-- val
   |-- test
   |-- normalize_mean.npz
   |-- normalize_std.npz
   |-- lat.npy
   |-- lon.npy
```

### Training

To finetune ClimaX for global forecasting, use
```
python src/climax/global_forecast/train.py --config <path/to/config>
```
For example, to finetune ClimaX on 4 GPUs use
```bash
python src/climax/global_forecast/train.py --config configs/global_forecast_climax.yaml \
    --trainer.strategy=ddp --trainer.devices=4 \
    --trainer.max_epochs=50 \
    --data.root_dir=/data/1.40625deg \
    --data.predict_range=72 --data.out_variables=['geopotential_500','temperature_850','2m_temperature','10m_u_component_of_wind','10m_v_component_of_wind','relative_humidity_850','specific_humidity_850'] \
    --data.batch_size=8 \
    --model.pretrained_path='https://huggingface.co/tungnd/climax/resolve/main/1.40625deg.ckpt' \
    --model.lr=1e-5 --model.beta_1="0.9" --model.beta_2="0.99" \
    --model.weight_decay=1e-5 
```
To train ClimaX from scratch, set `--model.pretrained_path=""`.

## Regional Forecasting

### Data Preparation

We use the same ERA5 data as in global forecasting and extract the regional data on the fly during training. If you have already downloaded and preprocessed the data, you do not have to do it again.

### Training

To finetune ClimaX for regional forecasting, use
```
python src/climax/regional_forecast/train.py --config <path/to/config>
```
For example, to finetune ClimaX on MENA region using 4 GPUs, use
```bash
python src/climax/regional_forecast/train.py --config configs/regional_forecast_climax.yaml \
    --trainer.strategy=ddp --trainer.devices=4 \
    --trainer.max_epochs=50 \
    --data.root_dir=/data/1.40625deg \
    --data.region="MENAreg" \
    --data.predict_range=72 \
    --data.out_variables=['geopotential_500','temperature_850','2m_temperature','10m_u_component_of_wind','10m_v_component_of_wind','relative_humidity_850','specific_humidity_850'] \
    --data.batch_size=8 \
    --model.pretrained_path='https://huggingface.co/tungnd/climax/resolve/main/1.40625deg.ckpt' \
    --model.lr=1e-5 --model.beta_1="0.9" --model.beta_2="0.99" \
    --model.weight_decay=1e-5 
```
To train ClimaX from scratch, set `--model.pretrained_path=""`.


## Citation

Please cite the following, if you find this work useful in your research:

```bibtex

```

## Contact
In case of any query, create issue or contact akhtar.munir@mbzuai.ac.ae 

## Acknowledgement
This codebase is built on <a href="https://github.com/microsoft/ClimaX">ClimaX</a>


