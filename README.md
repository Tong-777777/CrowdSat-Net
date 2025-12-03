# CrowdSat-Net

> "Crowd Detection (CD) Using Very-Fine-Resolution Satellites Imagery"

### Intorduction

- To the best of our knowledge, this is the first research to utilize VFR satellite imagery for CD, which aims to facilitate studies on the characterization of human spatial distributions and temporal activities at a large-scale (both spatially and temporally).
- To achieve this task, a novel CD dataset, CrowdSat, was collected by multi-source satellite platforms, which comprises over 120k labeled individuals and consists of diverse regions with strong heterogeneity, facilitating the development of CD methods. Additionally, during labeling, a point-like background removal strategy was introduced that uses multi-temporal VFR satellite imagery as auxiliary data to reduce mislabeling rates.
- A novel point-based CD method using the CNN, termed, CrowdSat-Net, was proposed to detect individuals in satellite imagery efficiently. Additionally, two innovational modules, DCPAN and HFGDU, were introduced to enhance individual feature presentation and recover the lost high-frequency information of individual features during upsampling.

### Data Source
![](./figures/data_spatial_distribution.png)
> Spatial distribution map of collected VFR satellite imagery for some locations where crowds typically gather.

To ensure broad spatial coverage and diverse conditions, we selected imagery spanning 32 provincial-level divisions in China (except Guizhou Province and Macao, due to fewer satellite images in these areas). These regions exhibit substantial heterogeneity, encompassing various landscapes and urban settings. These scenes include open areas (e.g., the Forbidden City), built-up areas (e.g., Kashgar People's Square), snowy regions (e.g., Harbin Ice and Snow World), areas with lush vegetation (e.g., the Emperor Qinshihuang's Mausoleum Site Museum), beaches (e.g., Qingdao Jiaozhou Bay National Marine Park) and desert regions (e.g., Yuesha Island). All the imagery was acquired between Feb. 20, 2023, and Jan. 2, 2025.

### CrowSat-Net
![](./figures/Hourglass.png)

> Overview of the proposed CrowdSat-Net. First, the labeled image is transformed into the FIDT map. During each training iteration, CrowdSat-Net enhances the basic features in the image preprocessing using the DCPAN module. Then, these enhanced features pass through the two-stacked Hourglass Networks. In each Hourglass Network, the traditional upsampling method is replaced with the HFGDU module to recover the lost fine details. Each Hourglass Network generates a location map, which is compared with the FIDT map to calculate the Focal Loss. The total loss L is the sum of Focal Losses from each Hourglass Network. After training, the location map conducted by the last Hourglass Network is transformed into the actual localization result using the LMDS method.

### Performance
Localization performance on the CrowdSat dataset.

|     Methods      | F1-score (%) | Recall (%) | Precision (%) |
|:----------------:|:------------:|:----------:|:-------------:|
|     SCALNET      |    60.38     |   51.86    |     72.26     |
|      P2PNet      |    49.60     |   49.17    |     50.04     |
|       PET        |    11.36     |   12.54    |     10.41     |
|     FIDTMCL      |    64.41     |   60.78    |     70.80     |
|      APGCC       |    64.34     | **63.05**  |     65.70     |
| **CrowdSat-Net** |  **66.12**   |   60.27    |   **73.23**   |

## Running CrowdSat-Net
### Prepare datasets
Please download the CrowdSat dataset, and set up the path to them properly in the configuration files.
- CrowdSat: the dataset will be available soon.
> Notes: There are two versions, one is the original dataset, the other is the enhancement dataset by three augmentation methods, i.e., horizontal flipping, vertical flipping and CutMix. In this experiment, we used the enhancement dataset.

Then, using the **./scripts/generate_list.py** to generate the corresponding list files, which is adopted as follows:

```
├── ./data
    ├── crowdsat
        ├── crowd_train.list
        ├── crowd_val.list
```

For each **.list** file, it should be adopted as follows:
```
E:\XXX\1.png E:\LLL\1.txt
E:\XXX\10.png E:\LLL\10.txt
...
```

### Prepare running Envs

See **requirements.txt**. 

### Ready to Run

Basically, you are recommanded to config the experimental runnings in a ".yaml" file firstly. 
You should change the **work_dir** in your **.yaml** file.

```shell
# train directly
python ./trainval.py

# fast test (you should change the checkpoint_path).
python ./demo.py

```
Before running the **demo.py**, please download the [checkpoint](https://drive.google.com/drive/folders/1ePFIN8bqG3ae07tXZxVVnTs5-arV3Aej?usp=drive_link).

## Citation

If you find these projects useful, please consider citing:

```bibtex
@misc{xiao2025crowdsat,
      title={Crowd Detection Using Very-Fine-Resolution Satellite Imagery}, 
      author={Tong Xiao and Qunming Wang and Ping Lu and Tenghai Huang and Xiaohua Tong and Peter M. Atkinson},
      year={2025},
      eprint={2504.19546},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.19546}, 
}

```
