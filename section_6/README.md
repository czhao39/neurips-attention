This folder contains the code for the ANN recognition experiments corresponding to Section 6 in the paper.

### Data preparation

We consider two ways of preparing the data for ANN recognition:
1. For a baseline condition, we generate masked images using saliency masks **from ANNs** in the notebook [`create_ANN_recognition_data_ANN_masks.ipynb`](create_ANN_recognition_data_ANN_masks.ipynb).
2. For the main condition of interest, we generate masked images using saliency masks **from humans** in the notebook [`create_ANN_recognition_data_human_masks.ipynb`](create_ANN_recognition_data_human_masks.ipynb).

### Experiment

The script [`run_nn_recognition.sh`](run_nn_recognition.sh) collects predictions from several models on the masked data.
Pretrained models can be downloaded via the instructions in the main README at ["Pre-trained Models"](../README.md#pre-trained-models); the directory containing the pretrained models can be pointed to with the variable `MODEL_DIR`.

### Results analysis

The notebooks [`analyze_ANN_recognition_ANN_masks_results.ipynb`](analyze_ANN_recognition_ANN_masks_results.ipynb) and 
[`analyze_ANN_recognition_human_masks_results.ipynb`](analyze_ANN_recognition_human_masks_results.ipynb) analyze results from masking with ANN and human masks, respectively.
Finally, the `R` script [`plot_recognition_results.R`](plot_recognition_results.R) generate plots that are equivalent to Figures 5 and S7 in the paper.
