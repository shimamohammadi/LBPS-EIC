# Uncertainty-driven-Sampling-for-Efficient-Pairwise-Comparison-Subjective-Assessment
Shima Mohammadi, João Ascenso

Instituto de Telecomunicações, Instituto Superior Técnico, University of Lisbon

[Download](https://arxiv.org/abs/2411.18372) the paper
## About the Paper
<p align="justify">
Assessing image quality is crucial in image processing tasks such as compression, super-resolution, and denoising. While subjective assessments involving human evaluators provide the most accurate quality scores, they are impractical for largescale or continuous evaluations due to their high cost and time requirements. Pairwise comparison subjective assessment tests, which rank image pairs instead of assigning scores, offer more reliability and accuracy but require numerous comparisons, leading to high costs. Although objective quality metrics are more efficient, they lack the precision of subjective tests, which are essential for benchmarking and training learningbased quality metrics. This paper proposes an uncertainty-based sampling method to optimize the pairwise comparison subjective assessment process. By utilizing deep learning models to estimate human preferences and identify pairs that need human labeling, the approach reduces the number of required comparisons while maintaining high accuracy. The key contributions include modeling uncertainty for accurate preference predictions and for pairwise sampling. The experimental results demonstrate superior performance of the proposed approach compared to traditional active sampling methods.
</p>

## How to Set Up

##### 1. Add Your Dataset:

- **Reference Images:** Place your reference images in the following directory:
    data/dataset/DATASET_NAME/REFERENCE_IMG_DIRECTORY
- **Distorted Images:** Organize your distorted images by source and place them in the following directory:
    data/dataset/DATASET_NAME/TEST_IMG_DIRECTORY

##### 2. Configure Variables:
In the main file (`src/main.py`), configure the following variables to match your setup:

- **`dis_dir`:** Path to the distorted images directory.
- **`ref_dir`:** Path to the reference images directory.
- **`triplets_names`:** Path to a CSV file containing all the triplets. An example CSV file is provided in the `dataset` directory.
- **`Defer_count`:** Specify the number of pairs you wish to defer to the subjective test.
- **`Conditions`:** Number of distorted images per source.

##### 3. Set Python Path:
Navigate to the project directory and add the `src` directory to your `PYTHONPATH` using the following command:
```
export PYTHONPATH=${PWD}
```
<!-- ##### 4. Download the Model:
- **[Download](https://drive.google.com/file/d/1WiVWDnS8IXkUzkVsC9uFHLmpxsQ4YcHd/view?usp=drive_link)** the trained model to src/model directory. -->

## How to Run the code
To execute the program, run the following command:

```
python src/main.py
```

## Citation
If you use this code in your research or projects, please cite the following paper:
```
@misc{mohammadi2024uncertaintydrivensamplingefficientpairwise,
      title={Uncertainty-driven Sampling for Efficient Pairwise Comparison Subjective Assessment}, 
      author={Shima Mohammadi and João Ascenso},
      year={2024},
      eprint={2411.18372},
      archivePrefix={arXiv},
      primaryClass={cs.MM},
      url={https://arxiv.org/abs/2411.18372}, 
}
```

## Acknowledgements

This work is funded by FCT/MCTES through national funds and when applicable co-funded EU funds under the project DARING with reference PTDC/EEI-COM/7775/2020.



