# AMDNloc
This repository is the Python implementation of paper _"[Multi-Sources Fusion Learning for Multi-Points NLOS Localization in OFDM System](https://ieeexplore.ieee.org/abstract/document/10669088)"_, which has been accepted by _IEEE Journal of Selected Topics in Signal Processing 2024_.

A simplified version, titled _"[Multi -Sources Information Fusion Learning for Multi-Points NLOS Localization](https://ieeexplore.ieee.org/abstract/document/10683036)"_ has been accepted for _2024 IEEE 99th Vehicular Technology Conference (VTC2024-Spring)_.


## Guidance
In this notebook, you will:

- Understand all the steps and frameworks of AMDNLOC.
- Follow and execute each line of code to generate data, process data, train neural networks, modify models during training, compare results, and more.
- After understanding this program, you can identify areas for improvement and enhancement. I warmly welcome anyone who needs to use my code, read my articles, and cite my work in their own research.
- All steps are contained within `compute_offset_WAIRD.ipynb`, acting as the main function. Open the program and execute it according to the outlined steps.
- If you have ideas for further improvements or new directions for enhancement, feel free to contact me as bohaowang@zju.edu.cn. We can discuss and collaborate to create more meaningful and valuable contributions. I am very eager to continuously improve and refine my work and code!
- If you find this helpful, please star this on GitHub 🌟🌟. Thanks so much!


## Table of Contents
* [Background Information](#background-information)
* [Program Execution Order](#program-execution-order)
* [Reference](#reference)
* [Dataset Generation](#dataset-generation)
* [AMDNLOC Multi-sources Framework](#amdnloc-multi-sources-framework)
* [Training and Testing Datasets Generation](#training-and-testing-datasets-generation)
* [Deep Learning Network Training](#deep-learning-network-training)
* [Some Visulization](#some-visulization)

## Background Information

The development of MIMO-OFDM enables the unique characterization of each point in a scene through multi-path effects. Fingerprint-based localization methods utilize channel information that reflects multi-path features, such as received signal strength (RSS), channel impulse response (CIR), channel frequency response (CFR), and angle-delay channel amplitude matrix (ADCAM), as fingerprints to infer coordinates. However, current fingerprint localization algorithms typically select a single type of channel information as the fingerprint, failing to fully capture the distinctiveness of each location. This leads to situations where distant points within a scene exhibit nearly identical fingerprints, thereby violating the assumption of independent and identically distributed (i.i.d.) data and significantly degrading localization accuracy.

To address this issue, conventional methods often manually partition the region into regular grids (e.g., rectangular grids) and iteratively reduce the grid size until the fingerprint that best matches the test sample is found. However, such regular division methods are inadequate for complex real-world scenarios, especially in the presence of irregular physical structures, such as buildings and scatterers, where the multi-path effects of samples within the grid may not exhibit high similarity. Furthermore, while existing irregular grid division methods have improved performance and accuracy, generating irregular grids often incurs high computational costs and fails to ensure consistent similarity among fingerprints within a grid.

Additionally, existing inference methods primarily establish a direct mapping between fingerprints and coordinates, overlooking issues such as regional covariates and heterogeneous data distributions during the inference process. These methods perform poorly in scenarios where fingerprints of neighboring locations exhibit high similarity and are prone to cumulative errors when the initial positioning is inaccurate. Therefore, there is an urgent need to further enhance AI-driven fingerprint localization algorithms to effectively address the complexity and diversity of real-world scenarios, improving localization accuracy and robustness.

## Program Execution Order

All steps are contained within `compute_offset_WAIRD.ipynb`, acting as the main function. Open the program and execute it according to the outlined steps.

1. Data Preprocessing

    a. Refer to [Dataset Generation](#dataset-generation) to generate the dataset.

    b. Run `gentarget_waird_git.py`.

    c. Run the 5th step of `templatematching_waird.py` in PyCharm.

    d. Follow the steps in `compute_offset_WAIRD.ipynb`.

2. Neural Network Training

    a. Modify the model by `sipml_shuai.py` in PyCharm.

    b. Run `train_CNN_shuai.py` in PyCharm to generate a CNN benchmark for training and comparing with our results.

    c. Run `train_prior_shuai.py` in PyCharm to produce the results processed by the AMDNLOC framework.

3. Visualization

    a. Run specific parts of `draw.ipynb` as a reference. 

    b. Write the code and draw whatever you need. 

p.s. Just run the code in order. Notes/reminders have already been written at the beginning of each cell. Chinese comments in the code won't affect the normal execution of the program.



## Reference
We welcome students and researchers who are working on academic research, thesis projects, or coursework to download and use our code. If you find this work helpful, **please give it a star** to support our efforts. Additionally, if you are following our research, **we kindly ask you to consider citing** the following papers:

- B. Wang, Z. Shuai, C. Huang, Q. Yang, Z. Yang, R. Jin, A. A. Hammadi, Z. Zhang, C. Yuen, and M. Debbah, "Multi-sources fusion learing for multi-points NLOS localization in OFDM system," lEEE J. Sel. Topics Signal Process., vol. 18, no. 7, pp.1339-1350, Sept. 2024, doi: 10.1109/JSTSP.2024.3453548.

- B. Wang, F. Zhu, M. Liu, C. Huang, Q. Yang, A. Alhammadi, Z. Zhang, and M. Debba, “Multi-sources information fusion learning for multi-points NLOS localization,” in Proc. Veh. Tech. Conf. (VTC), Jun. 2024, pp. 1–6.

For further inquiries or collaboration opportunities, feel free to contact me at: [bohaowang@zju.edu.cn](mailto:bohaowang@zju.edu.cn). I look forward to hearing from you!

```bibtex

@article{wang2024multiloc,
  title={Multi-Sources Fusion Learning for Multi-Points {NLOS} Localization in {OFDM} System},
  author={Wang, Bohao and Shuai, Zitao and Huang, Chongwen and Yang, Qianqian and Yang, Zhaohui and Jin, Richeng and Al Hammadi, Ahmed and Zhang, Zhaoyang and Yuen, Chau and Debbah, Mérouane},
  journal={IEEE J. Sel. Topics Signal Process.},
  year={2024},
  month={Sept.},
  volume={18},
  number={7},
  pages={1339 - 1350},
  doi={10.1109/JSTSP.2024.3453548},
  publisher={IEEE}
}

@INPROCEEDINGS{10683036,
  author={Wang, Bohao and Zhu, Fenghao and Liu, Mengbing and Huang, Chongwen and Yang, Qianqian and Alhammadi, Ahmed and Zhang, Zhaoyang and Debba, Mérouane},
  booktitle={Proc. Veh. Tech. Conf. (VTC)}, 
  title={Multi-Sources Information Fusion Learning for Multi-Points {NLOS} Localization}, 
  year={2024},
  month={Jun.},
  volume={},
  number={},
  pages={1-6},
  keywords={Location awareness;Wireless communication;Matched filters;Wireless sensor networks;Time-frequency analysis;Accuracy;Target tracking;Multi-sources;information fusion;fingerprint localization;inverse;heterogeneity;regional covariant},
  doi={10.1109/VTC2024-Spring62846.2024.10683036}}


```
All steps are contained within compute_offset_WAIRD.ipynb, acting as the main function. Open the program and execute it according to the outlined steps.
