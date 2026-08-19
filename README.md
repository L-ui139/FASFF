# FASFF for RGBT Tracking

Implementation of the paper [FASFF: Frequency Domain Information‑Aided Spatial Domain Feature Fusion for RGBT Tracking](https://www.sciencedirect.com/science/article/pii/S0893608026009585)


## Environment Installation
```
conda create -n fasff python=3.8
conda activate fasff
bash install.sh
```

## Project Paths Setup
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in `./data`. It should look like:
```
${PROJECT_ROOT}
  -- data
      -- lasher
          |-- trainingset
          |-- testingset
          |-- trainingsetList.txt
          |-- testingsetList.txt
          ...
```

## Training
Download [SOT](https://pan.baidu.com/s/1U42J6b3g1htma0OvmXRQCw?pwd=at5b) pretrained weights and put them under `$PROJECT_ROOT$/pretrained_models`.

```
python tracking/train.py --script tbsi_track --config vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --save_dir ./output/vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --mode single --nproc_per_node 1
```

Replace `--config` with the desired model config under `experiments/tbsi_track`.

## Evaluation
Put the checkpoint into `$PROJECT_ROOT$/output/config_name/...` or modify the checkpoint path in testing code.

```
python tracking/test.py tbsi_track vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --dataset_name lasher_test --threads 6 --num_gpus 1
```


### Results on LasHeR Testing Set


| Model | Backbone | Pretraining | Precision | NormPrec | Success | FPS | Checkpoint |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| FASFF | ViT‑Base | SOT | 71.2 | 67.1 | 56.7 | 41 | [BaiduYun](https://pan.baidu.com/s/1RRQV9yMVO6-hBjrKcRIEBQ), pwd: `3y2f` |

## Citation
If you find this work useful, please cite our paper:
```bibtex
@article{ZHANG2026109500,
title = {FASFF: Frequency Domain Information-Aided Spatial Domain Feature Fusion for RGBT Tracking},
journal = {Neural Networks},
pages = {109500},
year = {2026},
issn = {0893-6080},
doi = {10.1016/j.neunet.2026.109500},
url = {https://www.sciencedirect.com/science/article/pii/S0893608026009585},
author = {Jianming Zhang and Jun Long and Yonghang Liu and Lunfu Yu and Lei Wang},
keywords = {RGBT Tracking, Feature Fusion, Feature Enhancement, Spatial-Frequency Interaction, Discrete Cosine Transform},
abstract = {RGBT object tracking takes advantage of the complementary properties of RGB and thermal infrared (TIR) modalities. However, many existing methods focus on fusion within a single domain, either spatial or frequency, without fully exploiting the complementarity of multiple domains. This limits the interaction between domains and makes trackers less robust under severe conditions. To improve feature representation by leveraging both spatial and frequency information, we propose a Frequency domain information‑Aided Spatial domain Feature Fusion framework for RGBT tracking (FASFF), which mainly consists of the Frequency domain Attention Enhancement Module (FAEM) and the Spatial‑Frequency feature interaction Fusion Module (SFFM). The FAEM employs the Discrete Cosine Transform Attention block (DCTAttn), which highlights informative frequency components while integrating spatial and frequency domains information, and the Cross‑modality Collaborative Attention block (CCAttn), which suppresses noise via cross‑modal feature refinement to enhance intra‑modality feature representation. The SFFM further harnesses the interaction between the spatial and frequency domains to collaboratively refine feature representation, leveraging complementary information from the frequency domain to enhance spatial domain representations. We evaluate the FASFF on three benchmark datasets: RGBT210, RGBT234, and LasHeR. The experimental results show that the FASFF maintains stable and reliable performance across different tracking scenarios, verifying its effectiveness for RGBT object tracking.}
}
```

## Acknowledgments
Our project is developed upon [OSTrack](https://github.com/botaoye/OSTrack) and [TBSI](https://github.com/RyanHTR/TBSI). Thanks for their contributions which help us to quickly implement our ideas.

