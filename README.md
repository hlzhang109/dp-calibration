# A Closer Look at the Calibration of Differentially Private Learners

## Introduction
We systematically study the calibration of classifiers trained with differentially private stochastic gradient descent (DP-SGD) and observe miscalibration across a wide range of vision and language tasks. Our analysis identifies per-example gradient clipping in DP-SGD as a major cause of miscalibration, and we show that existing approaches for improving calibration with differential privacy only provide marginal improvements in calibration error while occasionally causing large degradations in accuracy. As a solution, we show that differentially private variants of post-processing calibration methods such as temperature scaling and Platt scaling are surprisingly effective and have negligible utility cost to the overall model.

Check our [paper](https://arxiv.org/abs/2210.08248) and [poster](https://github.com/hlzhang109/dp-calibration/blob/main/assets/dp_calibration_poster.pdf) at `assets/dp_calibration_poster.pdf `for more details

## Notes
- `train.sh` and `cv.sh` are used for training and inference. Once the logits are saved in npy files, use `calibration.py` to produce the recalibration results.
- Note that some NLI datasets' labels can have different (opposite) meanings (`label_mappings` in `classification/utils.py`). 

## Setup
- `pip install -r requirements.txt`

## Training
- Model training under various settings `bash scripts/train.sh`, `bash scripts/cv.sh` and `bash scripts/syn_exp.sh`.
- Recalibration python calibration/calibraiton.py.

## Citation

If you found this codebase useful in your research, please consider citing:

```
@misc{zhang2022closer,
      title={A Closer Look at the Calibration of Differentially Private Learners}, 
      author={Hanlin Zhang and Xuechen Li and Prithviraj Sen and Salim Roukos and Tatsunori Hashimoto},
      year={2022},
      eprint={2210.08248},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
