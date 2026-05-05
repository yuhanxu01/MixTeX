# MixTeX: Efficient LaTeX OCR with Synthetic Pretraining

MixTeX is a data-efficient LaTeX OCR system that converts scientific document images into editable LaTeX code. By introducing a novel synthetic pretraining approach that pairs Wikipedia text with LaTeX formulas, MixTeX achieves competitive performance with only **400 fine-tuning samples**, compared to existing methods requiring millions of real paired samples.

## Overview

**Key Features:**
- **Data Efficient**: Only 400 fine-tuning samples needed (vs. millions for baselines)
- **Cross-lingual**: Supports both English and Chinese scientific documents
- **Multi-format**: Handles printed and handwritten content
- **Synthetic Pretraining**: 120k synthetic pages from Wikipedia + 2M LaTeX formulas
- **Open Source**: Models, code, and datasets fully released

**Architecture:**
- **Encoder**: Swin Transformer (base size)
- **Decoder**: GPT-2 (medium size)
- **Training**: Two-stage pipeline (synthetic pretraining → real data fine-tuning)

**Test Datasets:**
| Dataset | Images | Language | Type | Size |
|---------|--------|----------|------|------|
| data_printed_chinese | 461 | Chinese | Printed | 7.8M |
| data_printed_english | 367 | English | Printed | 3.5M |
| data_handwritten_english | 81 | English | Handwritten | 2.5M |
| data_handwritten_chinese | 68 | Chinese | Handwritten | 1.3M |
| **Total** | **1,018** | - | - | **15.1M** |

## Installation

```bash
# Clone repository
git clone [https://github.com/yourusername/MixTeX.git](https://github.com/yourusername/MixTeX.git)
cd MixTeX

# Install dependencies
pip install -r requirements.txt

```

**Requirements:**

* Python >= 3.8
* PyTorch >= 1.12
* Transformers >= 4.20
* CUDA >= 11.0 (for GPU acceleration)

## Model

 **[Download Model Weights (Dropbox)](https://www.dropbox.com/scl/fo/f0tep888n743k1ed3yujf/AMJMoOnPjdj-nkzrXpRlPNY?rlkey=c3xm05bik7fdcfxixss9vs28i&st=lakyqwx6&dl=0)**

Please download the model files and place them in the `./model/` directory.

**Contents:**

* Model weights in HuggingFace style (`pytorch_model.bin`)
* Configuration files (`config.json`)
* Tokenizer files

**Load from local:**

```python
from transformers import AutoTokenizer, VisionEncoderDecoderModel, AutoImageProcessor

feature_extractor = AutoImageProcessor.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model", max_length=296)
model = VisionEncoderDecoderModel.from_pretrained("./model")

```

## Datasets

 **[Download Datasets (Dropbox)](https://www.dropbox.com/scl/fo/q6za5m5e2bpr46vv1kbij/ALGnTIIG5WJgHh1nuTW4CLM?rlkey=2vwc97ccuehecnwctxcxrrj16&st=5vt44itt&dl=0)**

### Pretraining Data (Synthetic)

**Location:** `./pretrain_data/`

**Contents:**

* `en1.zip`, `en2.zip`, `en3.zip` - English Wikipedia text + LaTeX formulas
* `zh1.zip`, `zh2.zip`, `zh3.zip` - Chinese Wikipedia text + LaTeX formulas
* `./pretrain_data_gen_code/` - Data generation scripts

**Total:** ~120,000 synthetic document pages

### Test Data (Real Documents)

**Location:** `./test_data/`

**Dataset Format (each zip):**

```
data_X_*.zip
├── *.jpg (images)
└── ground_truth.csv  # Format: file_name,text

```

**ground_truth.csv Example:**

```csv
file_name,text
1724718624.jpg,"..."

```

**Data Source:** Scientific documents from arXiv and Chinese academic repositories. For more details, see `datasets/dataset_description.md`.

## Inference

### Python Usage

```python
from code.inference import MixTeXInference

# Initialize model (ensure ./model folder contains the downloaded weights)
inferencer = MixTeXInference(model_path="./model")

# Run inference on a single image
latex_code = inferencer.predict("path/to/image.jpg")
print(latex_code)

```

### Command Line Usage

```bash
# Single image
python code/inference.py --model ./model --image path/to/image.jpg

# Batch processing directory
python code/inference.py --model ./model --image_dir ./test_data/ --output results.json

```

## Evaluation

To evaluate the model against the ground truth CSVs:

```bash
python code/evaluate.py \
    --model ./model \
    --csv path/to/ground_truth.csv \
    --image_dir path/to/images/ \
    --output evaluation_results.json

```

## Training

### Train with HuggingFace Dataset

```bash
python code/train.py --mode huggingface --output_dir ./results

```

### Train with Local CSV Data

```bash
python code/train.py \
    --mode local \
    --csv path/to/ground_truth.csv \
    --image_dir path/to/images/ \
    --output_dir ./results

```

## File Structure

```
MixTeX/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── code/                          # Source code
│   ├── train.py                   # Training script
│   ├── inference.py               # Inference script
│   ├── evaluate.py                # Evaluation script
│   └── utils.py                   # Utility functions
│
├── model/                         # Pre-trained model weights (Download from link above)
│   ├── pytorch_model.bin          
│   ├── config.json                
│   ├── tokenizer_config.json      
│   └── ...                        
│
├── test_data/                     # Test datasets (Download from link above)
│   ├── data_printed_chinese.zip
│   ├── data_printed_english.zip
│   ├── data_handwritten_english.zip
│   └── data_handwritten_chinese.zip
│
├── pretrain_data/                 # Synthetic pretraining data
│   ├── en1.zip
│   ├── en2.zip
│   ├── en3.zip
│   ├── zh1.zip
│   ├── zh2.zip
│   └── zh3.zip
│
└── pretrain_data_gen_code/        # Data generation code
    ├── gen.py
    ├── formular.tex
    └── endata1.txt

```
