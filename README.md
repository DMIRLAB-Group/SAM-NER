# Installation

The code is tested under python=3.10
pytorch=2.5.0 and CUDA Version Selected 11.8


## Dataset

[1] IEPile: https://github.com/zjunlp/IEPile

[2] PileNER: https://huggingface.co/datasets/Universal-NER/Pile-NER-type

[3] CrossNER: https://github.com/zliucr/CrossNER


## LLaMA-Factory
We use the LLaMA-Factory to training model and conduct inference.
[4] https://github.com/hiyouga/LLaMA-Factory/tree/main


# Project Structure

```text
SAM-NER/
├── data/
│   ├── crossNER/
│   │   ├── module1_ED/                  # Anchor Extractor data
│   │   │   ├── Llama3-8B/
│   │   │   │   └── predict/
│   │   │   └── test/
│   │   │
│   │   ├── module2_EE/                  # Explorer Extractor data
│   │   │   ├── Llama3-8B/
│   │   │   │   ├── predict/
│   │   │   │   └── predict_merge_tag/   # CCR -> tagged sentences
│   │   │   └── test/
│   │   │
│   │   ├── module3_EC/                  # Type Classifier data
│   │   │   ├── Llama3-8B/
│   │   │   │   └── predict/
│   │   │   └── test/
│   │   │
│   │   ├── module4_ETC/                 # Type Calibrator data
│   │   │   ├── Llama3-8B/
│   │   │   │   └── predict/
│   │   │   └── calibration/
│   │   │
│   │   └── schema/                      # Schema and schema definitions
│   │       ├── schema_abstract.json
│   │       ├── schema_crossner.json
│   │       ├── schema_define_crossNER.json
│   │       └── schema_define_IEPile.json
│   │
│   ├── eval/
│   │   └── Llama3-8B/                   # F1 score
│   │
│   ├── IEPile/
│   │   ├── module3_EC/
│   │   │   └── processed/               # Processed raw data
│   │   └── schema/                      # Schema mapping
│   │       └── schema_mapping.json
│   │
│   └── PileNER/
│       └── module2_EE/
│
├── datasets/                            # Raw datasets
│   ├── crossNER/
│   │   ├── gold/
│   │   └── raw/                         # Raw CrossNER datasets
│   │
│   ├── IEPile/                          # Download from [1] and rename
│   │                                      # `train.json` to `IEPile.json`
│   │
│   └── PileNER/                         # Raw Pile-NER-type dataset
│
├── models/
│   ├── EC/
│   ├── EE/
│   ├── iepile_lora/                     # [1] Llama3-8B LoRA
│   └── Meta-Llama-3-8B-Instruct/
│
├── src/                                 # Scripts
│   ├── EntityCalibration/
│   ├── EntityClassification/
│   ├── EntityDiscovery/
│   ├── EntityExplor/
│   ├── eval/                            # Evaluation scripts
│   ├── script/                          # Bash scripts
│   ├── yaml/                            # Model training & prediction configs
│   ├── crossNER_processor.py            # CrossNER raw data processor
│   └── tools.py
│
└── README.md
```

# Note: You need to verify whether the data or model paths mentioned in all the scripts are correct.

# Build Dataset
## Training Datasets
Create training dataset for `Explorer Extractor` and `Type Classifier`.

### EE
cd to directory `EntityExplor`, then run the `getTrainingData.py`. The training data is stored in `data/PileNER/module2_EE`. 

### EC
cd to directory `EntityClassification`, then run the `getTrainingData.py`. The training data is stored in `data/IEPile/module3_EC`.

## Test Datasets
Create test dataset for `Anchor Extractor` and `Explorer Extractor`.

First conduct `src/crossNER_processor.py` process the raw data of crossNER.

### Anchor Extractor
cd to directory `src/EntityDiscovery`, then run `getTestData.py`.The test data is stored in `data/crossNER/module1_ED/test`.

### Explorer Extractor
cd to directory `src/EntityExplor`, then run `getTestData.py`.The test data is stored in `data/crossNER/module2_EE/test`.


# Training model for EE and EC

## use Llamafactory train EE
cd to directory `src/yaml/EE`, then use `llamafactory-cli train` command to select `EE_lora_sft.yaml` for training the EE model.

## use Llamafactory train EC
cd to directory `src/yaml/EC`, then use `llamafactory-cli train` command to select `EC_lora_sft.yaml` for training the EC model.

# Use model conduct inference

## Anchor Extractor
cd to directory `src/script`, then conduct `ED_predict_and_get_result.sh`. The result of inference is stored in `data/crossNER/module1_ED/Llama3-8B/predict`.

## Explorer Extractor
cd to directory `src/script`, then conduct `EE_predict_and_get_result.sh`. The result of inference is stored in `data/crossNER/module2_EE/Llama3-8B/predict`.
And then cd to directory `src/EntityExplor`, conduct python script `merge_results.py`. Acquire based on E_final's tag sentence. The result of final is stored in `data/crossNER/module2_EE/Llama3-8B/predict_merge_tag`.

## Type Classifier
First create test dataset based on the results of `Anchor Extractor` and  `Explorer Extractor`:
cd to directory `src/EntityClassification`, then run `getTestData.py`.The test data is stored in `data/crossNER/module3_EC/test`.
And then:
cd to directory `src/script`, then conduct `EC_predict_and_get_result.sh`. The result of inference is stored in `data/crossNER/module3_EC/Llama3-8B/predict`.

## Type Calibrator
First create test dataset based on the result of `Type Classifier`:
cd to directory `src/EntityCalibration`, then run `getTestData.py`.The test data is stored in `data/crossNER/module4_ETC/calibration`.
And then:
cd to directory `src/script`, then conduct `ETC_predict_and_get_result.sh`. The result of inference is stored in `data/crossNER/module4_ETC/Llama3-8B/predict`.

# Evaluation
cd to `src/eval`, then conduct `eval.py`. The F1 score is stored in `data/eval/Llama3-8B`.

