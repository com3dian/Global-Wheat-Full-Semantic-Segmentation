# Global-Wheat-Full-Semantic-Segmentation

This repo is for the Wageningen team for the *Global Wheat Full Semantic Segmentation Challenge 2025*.

## Useful links
[Competition page](https://www.codabench.org/competitions/5905/#/pages-tab)

[Dataset Huggingface](https://huggingface.co/datasets/XIANG-Shuai/GWFSS-competition)

[Dataset Webpage](https://www.global-wheat.com/)

## Competition

### Timeline
🚀 Development Phase

🗓️ Duration: **March 15**, 2025 – **June 16**, 2025
Use this phase to tune your models and test against a small set of testing data.
Multiple submissions are allowed to refine your approach.

- [ ] 5 submission per day
- [ ] 100 submission total


🏆 Final Phase

🗓️ Duration: **June 16**, 2025 – **June 24**, 2025
Resubmit your best-performing model from the development phase.
Your model will be tested against a new set of testing data but trained on the same training data.
⚠️ Only *one submission* is allowed — choose wisely!

### Dataset
Splits: Pretraining, Train, Validation
- [x] *Pretraining data*: Over 64,000 images from 9 different domains, resolution: 512×512 pixels
- [x] Training data: Supervised fine-tuning data—99 images from 9 domains (11 images per domain), resolution: 512×512 pixels
- [x] Validation data: Used for model evaluation—99 images from 9 domains (11 images per domain), resolution: 512×512 pixels. Submit predictions to CondaBench to obtain mIoU scores.

```python
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("XIANG-Shuai/GWFSS-competition")
```

Related Works
See [here]()



