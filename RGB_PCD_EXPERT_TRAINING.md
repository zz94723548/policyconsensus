# RGB / PCD 专家：episode 划分与训练

本流程实现新路线的专家准备阶段：分别训练 RGB 与 PCD 动作扩散专家，为后续配对误差统计留出独立数据。PCD 是当前实验的第二模态；它不是触觉数据。

## 数据分配

源数据是 `data/rlbench/mt4_expert_200.zarr`，共 200 个完整 episode。按已确认的数据布局，每连续 50 个 episode 是一个任务。默认任务顺序取自原数据生成器的 `MT4_TASKS`：`open_box`、`toilet_seat_up`、`open_drawer`、`take_umbrella_out_of_umbrella_stand`。如实际名称顺序不同，通过划分命令的 `--tasks` 指定。

| 分区 | 每任务 | 总计 | 用途 |
| --- | ---: | ---: | --- |
| `expert_train` | 30 | 120 | 两个专家的梯度更新、动作/点云/状态归一化 |
| `expert_val` | 5 | 20 | 选择专家 checkpoint、调整专家超参数 |
| `risk_train` | 7 | 28 | 冻结专家后构造配对误差，训练后续风险统计模型 |
| `risk_calibration` | 4 | 16 | 后续统计量不确定集合的独立校准 |
| `test` | 4 | 16 | 最终独立评估；不用于选专家或风险模型 |

专家验证集不属于最终未触碰的测试集。专家训练与验证均不使用最后三个分区。两个专家使用同一组训练 episode，不把训练数据再分成 RGB 一半、PCD 一半。所有切分在 episode 层面完成；相邻帧和同一条轨迹的动作 chunk 不会跨分区。

这是现有 50 条/任务预算下的首版分配，校准和测试样本仍少。应先做全局统计与 episode 级区间，避免在每任务 4 条校准轨迹上声称细粒度条件覆盖。本流程从随机初始化重新训练。

旧配置实际设置了 `val_ratio=0.1, seed=42`，在 200 个 episode 中全局随机选择 180 条训练、20 条验证，并非全部 200 条都参与梯度更新。默认划分下四个任务的训练/验证数量依次为 46/4、48/2、42/8、44/6。但旧归一化使用全部 200 条数据，且旧划分没有给风险训练、校准和最终测试预留独立数据，因此旧权重不适合当前新划分的严格独立评估。

## 1. 固定划分

在仓库根目录、已激活项目环境的终端执行：

```bash
conda activate policy-consensus
python scripts/split_expert_data.py \
  --zarr data/rlbench/mt4_expert_200.zarr \
  --output splits/mt4_seed42.json \
  --seed 42 --episodes-per-task 50 --counts 30 5 7 4 4
```

仓库已为当前数据生成这份清单。相同参数重复执行会检查现有清单，拒绝覆盖内容不同的划分。如需比较另一分配，请写入新文件，并在两次专家训练中指定同一个新清单：

```bash
python scripts/split_expert_data.py --counts 25 5 10 5 5 \
  --output splits/mt4_25_5_10_5_5_seed42.json
# 训练命令加：data_split_manifest=splits/mt4_25_5_10_5_5_seed42.json
# 并通过 hydra.run.dir 指定新的输出目录。
```

清单包含 episode ID、episode 边界、任务块顺序、每任务分配及数据结构指纹。指纹检查边界与数组结构，不是全部传感器数据的内容校验和；划分后应固定源数据版本。加载器以只读方式按需访问 Zarr，不复制原始数据，不加载 DINO/PCA 特征。

## 2. 分别训练专家

```bash
# RGB：front / wrist / overhead 三路图像 + 本体状态
python scripts/train_modality_expert.py --config-name=train_rgb_expert

# PCD：fused_pointcloud XYZ + 同一组本体状态
python scripts/train_modality_expert.py --config-name=train_pcd_expert
```

两条命令可顺序执行。默认设备为 `cuda:0`，batch size 为 32，训练 1000 epochs，每 5 epochs 验证并保存 latest。默认 W&B 离线记录，不启动仿真。可追加 `training.device=cuda:1`、`training.num_epochs=500` 或 `dataloader.batch_size=16` 等 Hydra 参数。

两专家共用：动作维度 9、horizon 16、观测 2 步、执行前缀 8 步、DDPM 100 个时间步、相同 beta 日程、epsilon 预测和 U-Net 尺寸。动作/本体状态统计只拟合 `expert_train`，因此两专家的动作归一化相同。RGB 按已知 uint8 范围缩放为 `[0,1]`，再固定映射至 `[-1,1]`；PCD XYZ 按训练集统计归一化。RGB 专家不读取点云，PCD 专家不读取图像。

验证计算 EMA 专家的噪声预测 MSE，以固定验证随机种子减少 checkpoint 比较噪声。`best.ckpt` 由该损失选择；不使用任务成功率，也不生成模拟成功率指标。普通训练模型和 EMA 权重均保存在 MMPC 的标准 checkpoint payload 中。

## 3. 输出与继续训练

```text
output/experts/rgb_seed42/
  checkpoints/best.ckpt
  checkpoints/latest.ckpt
  split_manifest.json
  training_audit.json
  expert_config.yaml
  logs.json.txt
output/experts/pcd_seed42/
  ...
```

`training_audit.json` 保存训练/验证 episode 编号、保留分区、清单哈希、动作归一化参数及训练签名。后续配对评估使用各自 `best.ckpt` 中的 `ema_model`，并核对清单哈希、动作 scale/offset、噪声日程一致。风险模型使用 `risk_train`；不能用 `expert_train` 上的拟合误差代替 held-out 误差。

```bash
# 中断后按相同训练配置继续；已完成的运行会直接结束。
python scripts/train_modality_expert.py --config-name=train_rgb_expert training.resume=true
python scripts/train_modality_expert.py --config-name=train_pcd_expert training.resume=true
```

继续训练会检查数据/模型/训练签名并恢复优化器、学习率调度器、EMA 与 epoch。它不承诺与未中断运行逐位相同的随机数序列。更改超参数或划分后应使用新的 `hydra.run.dir`，避免混入已有 checkpoint。

## 4. 验证实现

```bash
python -m pytest tests/test_episode_split_experts.py -q

# 真实数据单步冒烟检查；缩小网络，产物写入 /tmp，不能用于正式结果。
python scripts/train_modality_expert.py --config-name=train_pcd_expert \
  hydra.run.dir=/tmp/pcd_expert_smoke \
  training.num_epochs=1 training.max_train_steps=1 training.max_val_steps=1 \
  training.lr_warmup_steps=0 \
  dataloader.batch_size=2 dataloader.num_workers=0 \
  val_dataloader.batch_size=2 val_dataloader.num_workers=0 \
  'policy.down_dims=[32,64]' policy.diffusion_step_embed_dim=32 logging.mode=disabled
```

测试覆盖分层划分重现、分区互斥、episode 边界、显式验证集、held-out 极端值不影响归一化、流式统计正确性、两个配置的扩散空间一致性及验证随机状态隔离。

## 5. checkpoint 保存策略与旧配置差异

当前训练入口只保留 `best.ckpt` 与 `latest.ckpt`：latest 每 5 epochs 覆盖，best 在 EMA 验证去噪损失改善时覆盖。没有保存每个阶段的独立 checkpoint，也没有继承旧入口的 Top-5 保留逻辑。已经完成的训练无法恢复未保存的中间权重。

相比仓库旧双专家默认配置，新入口还将 batch size 从 64 改为 32、diffusion timestep embedding 从 64 改为 128、验证间隔从 50 epochs 改为 5 epochs，并分别训练两个单模态模型。U-Net 通道 `[256,512,1024]`、DDPM 日程、动作/观测窗口一致。若旧 45 条模型沿用这些旧默认值，30/45 条模型的成功率对比同时包含训练配置差异，不能全部归因于数据量。

旧训练入口还有一个关键行为：`lazy_eval=True` 时不进行训练期仿真评估，而是把 `epoch / num_epochs` 写入 checkpoint 排序指标。旧默认排序指标虽命名为 `mean_success_rate`，在该分支中却是按 epoch 递增的占位值，所以 Top-5 会倾向于保留后期模型。这不是依据验证损失或真实任务成功率认定后期模型最好。新入口按 EMA 验证去噪损失最小值更新 best，两者选择标准不同。

旧专家阶段的总损失是两位专家去噪 MSE 的平均，验证调用未做 EMA 切换的 `self.model.compute_loss`；在开启 EMA 的旧流程中，切为 eval 的对象却是 EMA，因此旧验证仍可能使用训练模式下的随机 RGB 裁剪。新入口分别记录单专家 MSE，明确在 EMA 的 eval 模式下验证，RGB 使用中心裁剪，并固定验证噪声种子。两套验证数值不能直接当作相同测量口径比较。

本次实际日志显示，RGB 的最佳验证损失在 epoch 55（0.0082425），末轮为 0.0230929；PCD 最佳在 epoch 35（0.0107174），末轮为 0.0424265。两者在后期训练损失持续降低、验证损失持续升高，符合对当前训练轨迹过拟合的表现。每 epoch 有 509 次优化更新，两个最佳点分别已进行了 28,504 与 18,324 次更新。1000 epochs 是这次配置的训练预算，代码没有早停，因此仍继续训练到 epoch 999。该观察不证明相应 checkpoint 的仿真成功率最高。

## 6. 使用原仓库入口评测 latest 专家

录像沿用原 runner：`n_test_vis=8`，四个任务交错排列，因此前 8 个回合恰好每任务 2 个。每个专家一次完整评测新录制 8 条视频，RGB、PCD 合计 16 条，保存在各自输出目录的 `media/任务名/*.mp4`。重复使用同一输出目录可能留下旧视频，需要准确计数时使用新的输出目录。

`xvfb-run` 为后面的命令提供虚拟 X 显示服务，使 CoppeliaSim 可以在不弹窗口的情况下渲染观测和录像。`-a` 自动选择空闲显示编号；它不把测试转为后台运行，也不改变 `-d cuda:0` 指定的模型推理设备。

本次按用户指定，测试 RGB、PCD 的 `latest.ckpt`，使用其中的 EMA 权重。复用仓库已有的 `scripts/eval_policy.py sim`，不另写评测器。每个 checkpoint 的 `-n 80` 表示总计 80 回合，原 MT4 runner 轮流分配四个任务，因此每任务 20 次。RGB、PCD 各评测一次，总计 160 回合。

新训练 checkpoint 没有保存 `task.env_runner`，原评测入口为这些 checkpoint 复用已有 `task/rlbench/mt4.yaml` 的 runner 配置并禁用 DINO；仅加载模型和 EMA，不恢复训练优化器/学习率调度器。保留原 runner 的任务分配、动作执行和成功率计算流程。

```bash
cd /home/dbdx/NEUZZ/policyconsensus
conda activate policy-consensus
export PARALLEL_RLBENCH=1
export QT_QPA_PLATFORM=xcb

# 原项目 8 环境并行评测；前台运行，每专家总计 80 回合。
xvfb-run -a python -u scripts/eval_policy.py sim \
  -c output/experts/rgb_seed42/checkpoints/latest.ckpt \
  -o output/eval_experts_30_latest_parallel/rgb -n 80 -d cuda:0

xvfb-run -a python -u scripts/eval_policy.py sim \
  -c output/experts/pcd_seed42/checkpoints/latest.ckpt \
  -o output/eval_experts_30_latest_parallel/pcd -n 80 -d cuda:0
```

输出分别为 `output/eval_experts_30_latest_parallel/rgb/eval_log.json`、`output/eval_experts_30_latest_parallel/pcd/eval_log.json`。其中 `mean_success_rate` 是总体成功率，各任务的 `任务名/mean_success_rate` 是该任务成功率。原入口在输出目录存在时询问覆盖；运行前应使用新的 RGB、PCD 输出子目录。

原 runner 没有逐回合断点恢复功能，也没有保证环境/扩散随机数的配对。这里保留这些已有行为；若旧 45 条评测使用不同设置，成功率仅能作描述性比较。本次测试的是最终训练版本，而非验证损失最低版本。

当前使用原项目并行 runner，`mt4.yaml` 中 `n_parallel_envs=8`；80 回合分为 10 批、每任务仍为 20 回合。原代码注明并行录像可能存在跨回合缓冲问题，录像准确性与串行模式不同。
