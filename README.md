# 使用 Miniforge 去安裝環境
* 作業系統：macOS
* 程式語言：python
* 工具: 請參考 `./requirements.txt`

## Steps

1. Download and install Homebrew from https://brew.sh. Follow the steps it prompts you to go through after installed.
2. [Download Miniforge3](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh) (Conda installer) for macOS arm64 chips (M1, M1 Pro, M1 Max).
3. Install Miniforge3 into home directory.
4. Restart terminal.
5. Ｃhange directory.
```bash
cd ~ IMBD_data_competition_final_version   
```
6. Activate Conda environment.
```bash
conda activate ./env
```
7. Start Jupyter lab.
```bash
jupyter lab
```
8. 點進 Code 資料夾, 依序完整運行以下 notebook
* (可以跳過此步驟)01.EDA.ipynb 
* 02.feature_engineering.ipynb
* 03.model_catboost.ipynb
* 04.model_stacking.ipynb
* 05.model_predict.ipynb

9. 模型結果會產出在 `./Output/111921_TestResult.xlsx`