# nnU-Net Architecture Extractor

This tool extracts the deep learning architecture used by `nnU-Net` for a given configuration and saves it as a usable Python module.

## 🔍 What it does

- Reads a `nnUNetPlans.json` file (generated after running `nnUNet_plan_and_preprocess`).
- Identifies the architecture used (e.g., `PlainConvUNet`, `ResidualEncoderUNet`, etc.).
- Fetches the full class implementation from the official [dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures) GitHub repository.
- Extracts and filters only the required `import` statements used by the architecture.
- Downloads all dependencies (other `.py` files) required by the architecture.
- Generates:
  - `nnunet_extracted_architecture.py` containing the full class.
  - `test_architecture.py` to instantiate and test the model using the actual `arch_kwargs`.

## 📁 Output structure

All output is saved to a local `network/` folder:

## 🛠 How to use

1. Make sure your dataset has already been processed by `nnUNet_plan_and_preprocess`.
2. Run the script:

```bash
python ./main.py