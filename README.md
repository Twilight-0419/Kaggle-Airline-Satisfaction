# Predicting Airline Satisfaction

Kaggle `Predicting Airline Satisfaction`（Playground Series S6E10）的个人参赛项目。任务是根据乘客和航班信息预测满意度，本地评估指标为 ROC-AUC。

## 比赛成绩

- 最终 Private Leaderboard ROC-AUC：`0.96238`
- Public Leaderboard ROC-AUC：`0.96181`
- 最终排名：`222 / 3734`
- 最终提交文件：`submission_best_after_experiments.csv`

最终比赛成绩以 Private Leaderboard 为准。

## 当前结果

项目里保留了两类验证结果。严格本地验证与公开 OOF 堆叠需要分开理解。

| 实验 | 特征 | 平均折 AUC | OOF AUC |
|---|---|---:|---:|
| 280k 子集，LightGBM 原始特征 | 26 列 | 0.9587251 | - |
| 280k 子集，LightGBM + B20 | 28 列 | 0.9597156 | - |
| 全量开发集，LightGBM 原始特征 | 26 列 | 0.9589761 | 0.9589704 |
| 全量开发集，LightGBM + B20 | 28 列 | 0.9602089 | 0.9602000 |

B20 是两个平滑目标编码特征：

- `Flight Distance x Class`
- `Flight Distance x Type of Travel`
- 平滑参数为 20

完整开发集上的 B20 增益为 `+0.0012328 AUC`，五个折均取得提升。相应的摘要结果保存在 `outputs/research_run_001/`。

历史上还使用公开 OOF 预测做过堆叠，本地记录的最佳 OOF 约为 `0.962270465`。这部分依赖第三方公开预测，不能视为完全独立的本地验证结果。

## 目录结构

```text
.
├─ data/                 # Kaggle 数据，默认不提交到 Git
├─ notebook/             # EDA、主流程和历史 Notebook
├─ experiments/          # 特征实验、模型比较、堆叠和严格验证脚本
├─ outputs/              # 训练输出；仅少量摘要结果会提交
├─ make_best_submission.py
├─ make_submission_lb_candidate.py
├─ requirements.txt
└─ README.md
```

以下内容只保留在本地，没有上传到 GitHub：

- `data/` 中的训练集、测试集和提交文件
- `outputs/` 中的大型预测、矩阵和模型缓存
- `某个公开算法(Stacking TFMs with Public OOF)/` 第三方 Notebook、wheel 和本地依赖
- `catboost_info/` 训练缓存

## 环境

建议使用 Python 3.10 或更高版本：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

原始实验主要使用以下版本：

```text
pandas 2.3.3
scikit-learn 1.7.2
LightGBM 4.7.0
XGBoost 3.2.0
CatBoost 1.2.10
```

## 数据

从 Kaggle 比赛页面下载并放入 `data/`：

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

数据目录说明见 `data/README.md`。

## 运行严格验证

```powershell
python experiments/run_research_001.py audit
python experiments/run_research_001.py prepare
python experiments/run_research_001.py stage4 --models lightgbm
```

固定验证协议：

- 开发集：629,671 行
- 锁定验证集：69,964 行，占 10%
- 锁定随机种子：20261030
- 外层五折随机种子：42

锁定验证集只应在模型、特征、超参数和融合权重冻结后评估一次。

## 提交文件

历史最佳提交由以下命令重建：

```powershell
python make_best_submission.py
```

输出路径为 `data/submission.csv`。该脚本依赖未上传到 Git 的公开 OOF 文件和 `outputs/public_stack/` 中间结果，因此新克隆仓库时需要先准备这些本地依赖。

## 第三方材料

第三方公开 Notebook、预测文件和依赖包仅用于本地实验，目录已从 Git 中排除。使用或转载时请遵守原作者的许可和 Kaggle 比赛规则。
