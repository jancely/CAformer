# CAformer
TSLib is a code house for CAformer for deep time series forecasting.


## Usage

1. Install Python 3.8 or above. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well pre-processed datasets from `./data/`


3. Train and evaluate model. We provide the experiment scripts of our project under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
bash ./scripts/CAformer_ETTh1.sh
bash ./scripts/CAformer_ETTh2.sh
bash ./scripts/CAformer_ETTm1.sh
bash ./scripts/CAformer_ETTm2.sh
```

4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/CAformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.


5. Model Performance.
The forecasting performance of CAformer in ETT datasets is displayed as follows:

<p align="center">
<img src="./figures/performance.png"  alt="" align=center />
</p>


6. Ablation Study.
Further, we explored the effectiveness of Causal Attention (CA), we substitute CA with vanilla attention and conduct the experiment once again. The comparison of forecasting is exhibited as follows:

<p align="center">
<img src="./figures/ablation.png"  alt="" align=center />
</p>


7. Model Analysis.
We validated $Input Length$, $Model Efficiency$, and $Model Robustness$ as well. At last, we visualized the forecasting plot with ground-truth.


<p align="center">
<img src="./figures/input.jpg"  alt="" align=center />
</p>


<p align="center">
<img src="./figures/efficiency.png"  alt="" align=center />
</p>


<p align="center">
<img src="./figures/robustness.png"  alt="" align=center />
</p>


<p align="center">
<img src="./figures/visual.jpg"  alt="" align=center />
</p>



## Contact
If you have any questions or suggestions, feel free to contact:

- Chengli Zhou (chenglizhou@mail.ynu.edu.cn)

Or describe it in Issues.

## Acknowledgement

This research is supported by the National Natural Science Foundation of China (61862062, 61104035), \\
Yunnan Fundamental Research Projects (202401AT070471).

