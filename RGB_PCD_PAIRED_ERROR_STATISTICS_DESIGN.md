# RGB/PCD 配对误差与历史条件二阶统计模型设计

本文是实施设计与使用说明，不是已完成的实验报告。已固定数据子清单并实现方案 A 的离线误差采集、历史条件矩模型及训练入口（见第 12 节）；没有启动专家推理、风险训练或仿真，也没有修改已训练专家的配置和权重。

## 1. 本阶段目标

冻结当前两个 `latest.ckpt` 内的 EMA 专家，在 held-out 示范上构造共同噪声目标的配对误差，用每任务 5 条、总计 20 条 episode 训练历史条件模型，预测 RGB、PCD 误差的未中心化二阶统计量 A、B、C。每任务另 2 条、总计 8 条用于验证。

前面的训练阶段先验证点估计统计量，不训练成功率价值网络，不改变专家的模型或采样器。独立不确定性校准及鲁棒解析权重见第 14 节；闭环部署尚未接入。任务成功率是后续独立下游指标。

## 2. 固定数据及模型版本

原清单：`splits/mt4_seed42.json`。本阶段子清单：`splits/mt4_risk_model_seed142.json`。固定 seed=142，按原任务顺序，对每个任务原有 risk_train 的 7 个 episode 排序后随机置换，前 5 个训练、后 2 个验证。ID 为从 0 开始的全数据 episode 编号。

| 任务 | 统计模型训练 episode | 验证 episode |
| --- | --- | --- |
| open_box | 0, 6, 11, 12, 22 | 19, 35 |
| toilet_seat_up | 60, 72, 76, 81, 95 | 79, 82 |
| open_drawer | 100, 101, 118, 127, 141 | 110, 119 |
| take_umbrella_out_of_umbrella_stand | 167, 171, 173, 183, 194 | 154, 181 |

20/8 集互斥，合起来恰好覆盖原 risk_train 的 28 条，与 expert_train、expert_val、risk_calibration 和 test 均无重合。原专家训练集不重新划分；risk_calibration 的 16 条及 test 的 16 条不参与本阶段。

固定专家：

- `output/experts/rgb_seed42/checkpoints/latest.ckpt` 的 `ema_model`；
- `output/experts/pcd_seed42/checkpoints/latest.ckpt` 的 `ema_model`。

数据生成前记录两个文件的 SHA256、checkpoint epoch、配置、原清单哈希与子清单哈希，防止继续训练或替换文件后混用标签。直接实例化 checkpoint 保存的 policy，加载 EMA，设为 eval 并冻结全部参数；不调用训练 workspace.run()，不加载优化器或启动模拟器。

## 3. 对齐检查与动作窗口

复用 `DiffusionUnetPolicy`、`EpisodeSplitRlbenchDataset`、`SequenceSampler` 和既有动作归一化。提取当前两个模型的 obs_encoder 特征和 model 噪声预测，不调用各自 compute_loss() 分别生成随机噪声，否则无法保证配对。

必须检查：动作字段/物理顺序和单位、horizon=16、action_dim=9、n_obs_steps=2、n_action_steps=8、动作及共同本体状态的 normalizer scale/offset、DDPM betas/alphas_cumprod、epsilon 参数化、实际推理时间步完全相容。源数组 shape 与 checkpoint shape_meta 一致。实际数值以 checkpoint 为准，不能用当前 YAML 覆盖权重所对应的设置。

上下文从现有 sampler 的合法窗口中取，计划锚点为 buffer_start - sample_start + n_obs_steps - 1，按 episode 内每 8 个动作步选择一个锚点，模拟重新规划频率。严格复用 sampler 的 pad_before/pad_after 和动作窗口对齐，不额外构造超过其有效范围的末尾窗口。

风险模型和专家仅读取窗口前 2 帧观测；动作标签保留完整 16 步。起始观测重复首帧等规则与训练/部署一致。所有历史窗口限制在当前 episode 内。

## 4. 配对误差和监督标签

对上下文 h_t 及其示范动作 chunk a_0，用专家 checkpoint 的动作 normalizer 得到归一化动作。针对实际推理步集合 K（当前 100 步）中的每个 k，重复采样 M 个高斯噪声 epsilon_m，构造共同的 a_k：

    a_k = sqrt(alpha_bar_k) * a_0 + sqrt(1-alpha_bar_k) * epsilon_m
    pred_rgb = expert_rgb.model(a_k, k, global_cond=rgb_condition)
    pred_pcd = expert_pcd.model(a_k, k, global_cond=pcd_condition)
    e_rgb = pred_rgb - epsilon_m
    e_pcd = pred_pcd - epsilon_m

每个配对里的动作、时间步和噪声必须完全相同；两个专家只在观测条件上不同。RGB 使用 eval 模式的中心裁剪，特征提取关闭随机增强。

首版建议：100 个实际推理时间步全部覆盖，训练标签每步 M=4 次噪声、验证标签每步 M=8 次噪声。验证增加重复数只降低标签 Monte Carlo 方差，不增加独立 episode 数。重复数属于新离线模块的建议配置，不修改专家设置。

每个 k 的标签：

    A_tk = mean_m mean_(16x9) [e_rgb^2]
    B_tk = mean_m mean_(16x9) [e_pcd^2]
    C_tk = mean_m mean_(16x9) [e_rgb * e_pcd]

共享规划权重的统计目标：

    A_t = mean_k A_tk
    B_t = mean_k B_tk
    C_t = mean_k C_tk

因此 lambda_k=1/|K|，动作维度按 16x9=144 取均值，是一致的 epsilon-MSE 风险。它与平方范数求和相差固定倍数，不改变同一设置下最优权重。这里不是中心化方差/协方差，不减去误差均值；C 可以为负。

先做有限性、A>=0、B>=0、C^2<=AB（浮点容差）检查。共同噪声和共同归约下，样本矩本身应满足半正定性；明显违反时定位采集或对齐错误，不靠裁剪标签掩盖。

标签是固定示范动作下对扩散噪声/时间步求平均的样本估计，不是已知的真实条件期望。跨 episode 的监督回归用于估计 E[(A_t,B_t,C_t)|h_t]；单个 h_t 多采样噪声不能覆盖同一上下文下所有可能未来动作。

## 5. 存储与复现

建议按 episode 存储；先读取与缓存仅需的观测前缀，避免为每个 k 重复解码 RGB。一次计算观测编码，再复用于该上下文的全部时间步和噪声重复。

实际产物结构（执行采集命令后生成）：

    output/paired_errors/latest_seed2026/
      metadata.json
      risk_model_train/episode_XXXX.npz
      risk_model_val/episode_XXXX.npz

metadata.json 的 identity 中嵌入完整原清单和 20/8 子清单、专家 SHA256、采样日程和采集签名。

单 episode 文件保存：

- episode_id、task、anchor_frame、sequence_index、因果观测帧索引；
- rgb_condition [N,402]、pcd_condition [N,146]，维度需按实际 checkpoint 校验；
- timestep_ids [K]；moments_by_step [N,K,3]；moments [N,3]；
- paired_errors [N,K,M,2,16,9]，首版保存完整误差以审计 M1 和复算统计；
- 每个上下文/时间步/噪声重复的可重建 seed 规则。

seed 由固定主 seed、episode ID、anchor、k、repeat 和 train/val 标识确定；不要使用 Python 随进程变化的 hash()。采样不随批大小改变，便于验证结果复现。训练集和验证集使用独立噪声 seed。保留实际噪声日程和归一化摘要。

402=2*(3*64+7+2)，146=2*(64+7+2)：它们是现有专家的两帧全局条件向量，包含各自模态与本体状态，不包含动作、加噪动作、扩散噪声、未来观测或成功标签。

## 6. 风险统计模型输入与结构

首版采用方案 A：一次规划输出一组三元汇总统计量，跨整条去噪链复用。方案 A 不输入扩散时间步 k，也不逐 k 输出权重。方案 B 作为保留扩展，一次输出全部 K 个时间步的矩与权重；两方案均只在每次规划时调用一次统计模型，详见第 11 节。

首版上下文：当前与最近 3 次重新规划的冻结条件向量，即锚点 t、t-8、t-16、t-24。每个条件向量本身仅使用锚点及其前一帧。模型按从过去到当前的顺序处理有效历史；起始段仅有 1~3 个有效位置时，使用有效长度打包或显式 mask，不能把别的 episode 或未来特征补进来。

建议小模型：

    每个历史位置：
        RGB condition 402 -> Linear -> LayerNorm -> SiLU -> 64
        PCD condition 146 -> Linear -> LayerNorm -> SiLU -> 64
        concatenate -> 128
    有效历史序列 [L<=4,128] -> 1-layer GRU(hidden=64)
    最后有效隐状态 -> Linear(64,32) -> SiLU -> Linear(32,3)

第一版不微调专家编码器，不输入 task_id、episode ID 或归一化轨迹进度；任务名仅用于分层采样、划分和结果报告。相对历史间隔/有效 mask 可作为输入，需只来自当前/历史观测。

缓存特征如需标准化，仅用 20 条训练 episode 拟合；验证、校准和测试阶段使用固定训练统计。特征尺度过小的维度使用固定数值下界。

## 7. 保证矩合法性的输出头

令头部输出 u,v,r，训练集尺度 s=max(mean_train((A+B)/2),1e-8)，此均值按任务和 episode 均衡。s 仅在 20 条训练数据上拟合并写入 checkpoint。

    a = softplus(u) + 1e-6
    b = softplus(v) + 1e-6
    rho = tanh(r)
    A_hat = s*a
    B_hat = s*b
    C_hat = s*rho*sqrt(a*b)

由此 A_hat、B_hat 为正，且 C_hat^2 <= A_hat*B_hat；无需单独依赖 PSD 惩罚。不把 rho 当作中心化误差相关系数，它只是满足未中心化矩可行性的参数。

## 8. 损失、采样和 checkpoint

主训练损失是在原二阶矩尺度上的固定尺度平方回归：

    l_t = ((A_hat-A_t)/s)^2 + ((B_hat-B_t)/s)^2 + 2*((C_hat-C_t)/s)^2
    L = mean_task mean_episode mean_context l_t

这相当于归一化 2x2 矩阵的 Frobenius 平方误差。固定尺度平方损失对应条件均值估计；不对 log(A) 做回归后直接指数还原，也不按每个样本自身 A+B 除损失，以免改变本阶段统计目标。暂不加权重回归、成功率 BCE 或直接融合风险辅助项。

采样先均匀选任务，再均匀选该任务训练 episode，再均匀选上下文，避免长轨迹/高重叠窗口主导训练。训练和验证所有聚合均以 episode 为独立单位；不能把多噪声或相邻上下文作为独立轨迹计算置信区间。

建议的新风险模型训练配置（不是专家训练配置）：AdamW，lr=3e-4，weight_decay=1e-4，batch_size=64，gradient_clip_norm=1.0；最多 500 epochs，每 epoch 验证，默认关闭早停并跑满 500 epochs；early_stopping=true 时才启用 min_epochs=20、patience=30 的可选停止规则。无必要再引入风险网络 EMA。所有配置在实施前固定，不自行改变既有专家参数。

checkpoint 保存策略明确为：best（验证矩回归损失最低）、latest（每 epoch 覆盖）、每 25 epochs 保留一个带 epoch 编号的快照。缓存/划分哈希、模型结构、输入标准化、输出尺度 s、优化器与训练参数随 checkpoint 保存。中间快照属于明确计划的行为，不重复此前只留 best/latest 的隐式简化。

## 9. 从统计量验证权重，而非只看回归损失

用预测矩计算一次规划共享的 RGB 权重：

    w_rgb = clip((B_hat-C_hat)/(A_hat+B_hat-2*C_hat), 0, 1)
    w_pcd = 1-w_rgb

分母接近零时按预先固定的相对容差回退到训练集全局风险权重。全局风险权重也只由 20 条训练 episode 的矩均值确定。任何方法都不能用 8 条验证标签为每个样本选最优 w 再将结果作为可部署方法。

在相同的 8 条验证 episode、相同配对误差上比较：RGB 单专家、PCD 单专家、固定等权、训练集全局最小风险固定权重、无历史当前条件 MLP、带历史的统计模型。后两者都预测矩再求权重。先跑全局基线检查统计公式和数据，再训练条件模型。

使用验证真实配对误差或其样本矩计算：

    R_val(w) = w^2*A_val + (1-w)^2*B_val + 2*w*(1-w)*C_val

报告：总体/逐任务/逐扩散步风险，A/B/C 回归误差，矩合法率，权重分布与相邻规划抖动，fallback 比例。风险差按 episode 配对，并按任务分层 bootstrap；每任务仅 2 条验证轨迹，区间稳定性有限，不作强条件保证。

主 checkpoint 仍按事先指定的验证矩回归损失选择；融合风险是必须核查的效果指标。全局或条件权重在独立验证上不改善时，不直接进入复杂鲁棒模块。可用固定的额外噪声 seed 检查结论对 Monte Carlo 抽样的敏感性，但不得据最终 test 结果反复调参。

## 10. 实施顺序与边界

1. 审计 checkpoint、原数据清单和 20/8 子清单；通过后冻结版本。
2. 构造 28 条 episode 的配对误差与因果特征缓存，训练/验证角色固定。
3. 只用 20 条拟合全局 A/B/C/w，在 8 条验证公式、数据及基线风险。
4. 用相同缓存训练当前条件 MLP，再训练短历史模型，比较 held-out 风险。
5. 选定并冻结风险模型后，再进入独立 16 条 risk_calibration 的不确定性校准；最终 16 条 test 留到完整方案固定后。

本阶段不是仿真成功率评测，不需要 xvfb-run 或 RLBench 环境启动。已有专家推理、数据和 normalizer 接口直接复用；配对误差和统计模型属于路线新增能力，后续在其基础上增量实现离线模块，不另写通用仿真评测器。数据集/Numba 导入保留在离线训练运行函数中，避免再次污染原仿真评测的 LLVM 加载链。

所有结论首先限于 held-out 示范的前向加噪分布。条件噪声误差降低不自动推出反向采样分布上的误差或真实任务成功率改善。统计模型的不确定集合和部署支持检查不属于这一步已完成的功能。

## 11. 方案 A 先验证，方案 B 保留

### 11.1 当前阶段：方案 A（共享权重）

h_t 表示当前观测与因果历史。缓存先保留每步矩，再离线用固定 lambda_k>=0、sum(lambda_k)=1 汇总：

    A(h)=sum_k lambda_k*A_k(h)
    B(h)=sum_k lambda_k*B_k(h)
    C(h)=sum_k lambda_k*C_k(h)
    w(h)=clip((B-C)/(A+B-2*C),0,1)

训练与部署的输出头是 3 维。一次规划前向一次统计模型并解析一次，所有去噪步复用该权重；在线不读取真实误差，也不逐步累计 A/B/C。当前首版 lambda_k=1/K，验证共享权重的独立去噪风险及其与全局固定权重的差异。

### 11.2 保留扩展：方案 B（一次生成逐步权重）

共用因果特征、历史编码器和数据划分，将输出头从 Linear(32,3) 扩展为 Linear(32,3*K)，重排为 [batch,K,3]。每步采用相同的 PSD 结构化参数化，直接监督 moments_by_step：

    w_k(h)=clip((B_k-C_k)/(A_k+B_k-2*C_k),0,1)

每次规划只运行一次统计模型，随后在规划开始时批量解析 K 个权重，在去噪循环内查表使用。解析求权重发生在部署时；离线生成用于训练/分析的参考权重不代替在线根据新观测求权重。不同 k 不再汇总成一个共享权重。它依赖规划时的 h 和预定义噪声阶段，不依赖尚未产生的中间 a_k；后者属于额外的反馈路由扩展，不在本方案内。

接口约定（方案 A 的训练与验证已实现；方案 B 仅保留标签与设计）：

- risk_mode=shared：输出 moments [batch,3] 与规划权重 [batch,1]；
- risk_mode=stepwise：输出 moments [batch,K,3] 与规划权重 [batch,K]；
- timestep_ids 必须与专家实际采样器的时间步逐项对应，不能把数组索引当成扩散时间；
- 首阶段只训练 shared；不得静默修改已训练专家、采样步数或把 A 改成 B。

### 11.3 采集阶段现在必须保留的内容

moments_by_step [N,K,3]、paired_errors [N,K,M,2,16,9]、timestep_ids、固定 lambda_k、专家版本、条件特征和噪声 seed 规则均保留。汇总的 moments [N,3] 是由逐步矩导出的方案 A 标签，不能只存汇总后删除逐步标签。沿用同一批 20 条训练、8 条验证，B 无需更换数据划分或另行采集原始误差。

### 11.4 研究比较与理论边界

固定同一 h 和同一前向加噪评估分布，若真实矩已知，且 B 的各步权重没有额外耦合约束，则：

    sum_k lambda_k * min_w R_k(w|h)
        <= min_w sum_k lambda_k * R_k(w|h)

这是 B 包含共享权重 A 作为特例的优化空间关系。统计量需要估计时，该不等式不保证学习到的 B 优于 A；逐步目标的有限样本方差和模型容量可能抵消优势。若加入平滑/低秩等约束，需要重新检查可行集合关系。

在 8 条验证轨迹上可先做逐步参考最优权重分析，诊断相对误差随噪声阶段的变化及潜在收益；这是用验证标签得到的 oracle 诊断，不是可部署性能。A/B 的实际比较必须用各自仅在 20 条训练轨迹上学到的统计预测，报告相同验证标签上的风险、按噪声阶段的分解、推理延迟、输出参数量及矩预测误差。两者的统计模型调用次数均为每规划一次。

最优共享权重只能最小化所定义的累计风险，不必在每个 k 上同时优于两个专家。已有凸 score 组合及采样稳定性工作是理论参考，不直接保证本项目的学习权重、有限样本风险或任务成功率改善。参考：[GPC](https://arxiv.org/html/2510.01068v1)。

### 11.5 方案 B 的候选贡献与相关工作边界

候选贡献应围绕：历史条件化、一次前向预测整条噪声阶段的配对误差二阶矩、解析逐步融合，以及后续不确定性校准；具体新颖性需与已有机器人策略组合及生成模型组合方法逐项比较。

暂不宣称“首次在去噪步骤使用时变组合”。已有生成模型组合研究讨论随采样过程变化的专家权重，例如 [FactorDiff](https://arxiv.org/html/2607.11758v1) 对 SuperDiff、RNE 等方法的讨论。它们与此处冻结模态专家、条件误差风险监督和一次预测整组权重并非同一设定，但足以说明广义时变组合不是可以直接认定的空白。保留 B 的研究价值，不预先认定其优越性或首创性。

## 12. 已实现入口与运行命令

新增能力通过已有 `scripts/manip_policy.py` 的 Hydra 入口运行；原专家训练、原仿真评测均未修改。采集/训练都是离线程序，不需要 `xvfb-run`，也不创建 RLBench 环境。

### 12.1 构造配对误差缓存

```bash
cd /home/dbdx/NEUZZ/policyconsensus
conda activate policy-consensus
python -u scripts/manip_policy.py --config-name=collect_error_moments
```

配置：`modular_policy/config/collect_error_moments.yaml`。默认冻结两个 latest.ckpt 的 EMA，100 个现有推理时间步，规划间隔 8，训练噪声重复 4、验证重复 8，采集 batch size=16。默认保存完整逐步配对误差与条件特征，路径为 `output/paired_errors/latest_seed2026/`。

每完成一个 episode 都写入原子文件和 SHA256 记录。相同专家、代码、划分与采集设置下，中断可继续：

```bash
python -u scripts/manip_policy.py --config-name=collect_error_moments resume=true
```

batch_size 可调整而不改变上下文所用噪声，例如追加 `batch_size=8`。更换专家、噪声重复或其他数据定义需使用新的 `cache_dir`。采集日志按 episode 显示进度；一个 episode 内会计算所有选定上下文、全部时间步及多次噪声，进度条不会逐次网络调用更新。

### 12.2 训练方案 A 的二阶矩模型

只有采集完成、metadata.status=complete 后才能训练：

```bash
python -u scripts/manip_policy.py --config-name=train_error_moments
```

配置：`modular_policy/config/train_error_moments.yaml`。使用固定 20/8 轨迹；训练集独立拟合特征标准化和输出尺度。4 个历史位置、双 64 维投影、GRU hidden=64、32 维隐藏头和三个 PSD 参数。模型输出 A/B/C，解析权重在验证中计算；尚未将风险模型接入真实/仿真在线控制循环。

AdamW lr=3e-4、weight_decay=1e-4、batch size=64、梯度裁剪 1；最多 500 epochs，每轮验证。默认 training.early_stopping=false，完整运行 500 epochs；min_epochs=20、patience=30 仅在显式开启早停时生效。风险网络不另加 EMA。上述均为新增风险模型参数，不是专家参数改动。

默认输出：

    output/error_moments/shared_history4_seed42/
      baselines.json
      best_validation.json
      best_validation_predictions.npz
      training_audit.json
      cache_metadata.json
      risk_config.yaml
      logs.json.txt
      checkpoints/best.ckpt
      checkpoints/latest.ckpt
      checkpoints/epoch_0025.ckpt
      checkpoints/epoch_0050.ckpt
      ...

每轮更新 latest；按验证矩回归损失保存 best；每完成 25 epochs 保留独立阶段快照。早停前未到达的阶段不会产生对应快照。继续相同训练：

```bash
python -u scripts/manip_policy.py --config-name=train_error_moments resume=true
```

数据/模型/代码/训练签名不一致时拒绝恢复，需使用新输出目录。保存优化器、epoch、best 指标和均衡采样随机状态。

### 12.3 查看结果

baselines.json 给出训练集全局权重及其在 8 条验证轨迹上的 RGB/PCD/等权/全局权重风险。best_validation.json 额外报告条件统计权重的逐任务、逐 episode、逐扩散步风险、原尺度矩 MSE、权重范围、回退比例和相邻规划权重变化；差值区间使用按任务分层的 episode bootstrap，仅 8 条验证轨迹，应谨慎解读。

主 checkpoint 只按归一化矩阵 Frobenius 回归损失选择，不按成功率或事后挑选的融合风险选择。PSD 合法率基于结构化输出和有限数值容差检查；有非有限值或明显非法矩时停止。

方案 B 目前没有训练/部署实现。采集缓存保留了其需要的 moments_by_step、timestep_ids 与完整误差；`risk_mode` 仅接受 shared，设置为其他值会明确报错，不会悄悄替换方案。

### 12.4 代码检查

```bash
python -m pytest tests/test_error_moments.py -q
```

这些检查只用合成矩阵、随机噪声与临时缓存验证代数、PSD、20/8 隔离、因果历史、归一化与完整性；不加载真实专家、不执行真实模型推理/训练或启动仿真。真实采集与训练仍由用户按上述命令运行。

## 13. 固定预算训练诊断（取消默认早停）

两次初始训练分别在 60 和 39 epochs 停止，每 epoch 只有 6 次参数更新，无法从这些短曲线确认后期不会改善。当前统计模型是监督矩回归网络，不是扩散去噪专家；专家的去噪 MSE、矩回归误差、融合风险与机器人成功率是不同指标，不能根据其中一个自动保证另一个最优。

按用户要求，默认关闭早停；两种 history 配置按相同 500 epochs、相同缓存和原有网络/优化器设置重新训练。每轮记录：

- train_moment_loss：epoch 结束后，在全部训练上下文上以 episode 均衡方式确定性计算；
- sampled_train_moment_loss：原有放回采样训练批次的平均损失，可能因长尾样本抽样波动；
- val_moment_loss：验证矩回归误差；
- train_fusion_risk、val_fusion_risk：使用预测统计量解析权重后，在相应缓存上的融合 epsilon MSE。

best.ckpt 与 best_validation.json 继续对应矩回归损失最小版本。新增 best_risk.ckpt 与 best_risk_validation.json，单独保存验证融合风险最低版本。latest.ckpt 与 latest_validation.json 对应最后完成的 epoch；每 25 epochs 的阶段快照继续保留。run_summary.json 保存实际训练预算完成情况。

关闭早停用于排查训练预算与选模指标的影响，不预先声称最后一轮更好。8 条验证轨迹用于训练诊断和选模，其 bootstrap 区间不构成最终独立检验；16 条校准与16条最终测试继续保留。

### 13.1 本次固定预算重训结果

history=4 和 history=1 均已完成 500 epochs，保留 25~500 每 25 epochs 的快照。汇总见 output/error_moments/full_budget_comparison.json。

| 历史长度 | 矩损失最佳 epoch / 融合风险 | 融合风险最佳 epoch / 融合风险 | 最后 epoch / 融合风险 |
| --- | --- | --- | --- |
| 4 | 129 / 0.00248918 | 24 / 0.00235012 | 499 / 0.00264897 |
| 1 | 8 / 0.00278429 | 30 / 0.00234115 | 499 / 0.00268411 |

关闭早停后，history=4 的矩回归误差在 epoch 129 进一步改善；但两模型最低融合风险都出现在原停止位置之前。因此这次最明确的问题是矩回归指标与融合风险的选模差异，而不是已经证明后期融合更好。best_risk.ckpt 按开发验证风险选择，不能当作未触碰最终测试集的结果。两种历史配置在该指标上接近，不能据此确认额外历史的稳定优势。

## 14. 冻结方案 A 后进行独立校准

当前固定主模型为 `shared_history4_seed42/checkpoints/best_risk.ckpt`，history=1 留作消融。下面两个命令由用户执行；代码检查没有运行实际采集、校准或仿真。校准不会更新参数、重新选 checkpoint 或使用最终 test 的 16 条轨迹。

```bash
cd /home/dbdx/NEUZZ/policyconsensus
conda activate policy-consensus

# 先等待 16 条独立校准轨迹采集完成；复用原配对误差采集器
python -u scripts/manip_policy.py --config-name=collect_error_moments_calibration

# 冻结模型，计算校准半径和鲁棒权重诊断
python -u scripts/manip_policy.py --config-name=calibrate_error_moments
```

采集中断后，用第一条命令加 `resume=true` 恢复。校准缓存位于 `output/paired_errors/calibration_latest_seed2026`，不会覆盖原来的 20/8 训练验证缓存。沿用同一对专家、100 个噪声时间步及均匀汇总、每 8 个动作时刻一个上下文；每时间步使用 8 次独立 Monte Carlo 噪声，与冻结验证协议一致。代码校验专家文件哈希、数据划分、特征和标签协议。

对上下文的预测矩阵 Gamma_hat 和 Monte Carlo 标签 Gamma_MC，定义分数为 `||Gamma_MC-Gamma_hat||_F / s`，其中 s 是训练时固定的输出尺度；一整条 episode 的分数取全部采样上下文分数的最大值。完整轨迹是校准单位，不能把相关的时间窗口当成独立校准样本。

每任务只有 4 条校准 episode，默认 alpha=0.2，有限样本顺序统计量位置为 `ceil((4+1)*0.8)=4`。先取各任务的分位数，再取四个任务阈值的最大值作为共享半径。这样无需部署时读取任务标签，也不假定不同任务之间可交换。在各已知任务内部轨迹可交换、模型及采集协议预先冻结的假设下，对该任务新轨迹上的有限 Monte Carlo 标签提供至少 80% 的边际轨迹覆盖。这里不保证潜在真实条件二阶矩、部署分布偏移或任务成功率。90% 对应第 5 个顺序统计量，现有每任务 4 条数据无法得到有限阈值；本实现明确拒绝该设置。

不确定集合为 PSD 矩阵与 Frobenius 球的交集：`Gamma >= 0, ||Gamma-Gamma_hat||_F <= r`。令 RGB 权重为 w，其精确最坏风险为：

    R_hat(w) + r * (w^2 + (1-w)^2)

所以鲁棒解析权重为：

    w_robust = clip((B+r-C)/(A+B+2*r-2*C), 0, 1)

r=0 时退化为原点估计权重；r 增大时更接近等权。该风险上界仅在所关注的实际矩阵落入集合时成立。当前实现只产出校准文件和离线诊断，尚未改变项目原生仿真入口。

输出目录 `output/error_moments/calibration_shared_history4_seed42/`：

- `calibration.json`：冻结 checkpoint 哈希、16 条 episode 分数、共享半径、解析规则及适用假设；
- `calibration_diagnostics.json`：RGB、PCD、等权、全局固定权重、条件点估计、鲁棒权重的校准集离线风险；
- `calibration_predictions.npz`：预测矩、标签、权重、误差分数及轨迹索引。

校准集内覆盖率和风险是拟合诊断，不能作为独立泛化评估或用来反复调整 alpha、模型及集合形式。完成后先检查半径、权重范围和逐任务诊断，再基于项目原有评测流程接入方案 A；最终 test 仍需保留。

```bash
python -m pytest tests/test_error_moments.py tests/test_moment_calibration.py -q
```

共 20 项合成数据测试，覆盖原训练隔离、独立校准分区、协议审计、有限样本分位数和鲁棒解析解。有限样本校准依据见 [Conformal prediction: a unified review of theory and new challenges](https://academic.oup.com/jrsssb/article/87/4/1100/8058684)。

## 15. 通过原生 RLBench 入口对比方案 A

已在 `scripts/eval_policy.py sim` 增加 `--fusion equal|point|robust`、`--pcd-checkpoint`、`--calibration`、`--eval-seed` 参数，没有另写评测器。单专家原命令继续可用。模型训练配置、checkpoint、校准半径不变。

融合适配器调用原 `DiffusionUnetPolicy.conditional_sample`。每步两个专家读取相同的带噪动作，按权重合并 epsilon 预测，再执行一次原 scheduler.step；不会分别生成两个动作后再平均。每次规划编码一次各专家观测，point/robust 模式调用一次冻结统计模型；整条 100 步去噪链共用该权重。equal 模式不调用统计模型。历史按环境批次的独立通道保存，最多 4 个规划上下文，格式与训练一致；原 runner 每个新 episode 批次调用 reset 清空历史。

三组默认使用同一对 latest 专家的 EMA、同一个 history=4 best_risk 统计模型与校准产物。代码校验专家 SHA、统计 checkpoint SHA、源划分、动作尺度、扩散步骤和规划间隔。即使 equal 不使用统计模型，也绑定同一套实验来源。

```bash
cd /home/dbdx/NEUZZ/policyconsensus
conda activate policy-consensus
export PARALLEL_RLBENCH=1
export QT_QPA_PLATFORM=xcb

# 三组顺序前台运行；每组 80 次=4任务各20次，总计240次
for mode in equal point robust; do
  xvfb-run -a python -u scripts/eval_policy.py sim \
    -c output/experts/rgb_seed42/checkpoints/latest.ckpt \
    --pcd-checkpoint output/experts/pcd_seed42/checkpoints/latest.ckpt \
    --calibration output/error_moments/calibration_shared_history4_seed42/calibration.json \
    --fusion "$mode" --eval-seed 42 \
    -o "output/eval_fusion_scheme_a_seed42/$mode" -n 80 -d cuda:0 || break
done
```

沿用 MT4 配置：8 个并行环境、初始环境种子 1000、交错任务、每任务 20 次、视频总计 8 条（每任务 2 条）。三组保持相同并行数、episode 顺序与规划步数；扩散随机数使用独立 generator，种子为 `eval_seed + 批次编号`，从而避免构造统计模型消耗随机数而改变采样初始噪声。闭环状态和提前终止仍可能随策略而变化。旧单专家结果可作参考，但未使用这里的独立 generator，不应声称与新三组严格匹配扩散随机数。

各组目录保存：

- `eval_log.json`：总体/逐任务成功率，以及 `episode_results` 中每个环境 seed 对应的任务和成败（并行 runner）；
- `fusion_config.json`：融合模式、专家/统计模型/校准文件哈希、随机种子和权重顺序；
- `fusion_trace.json`：每个规划批次的 RGB 权重及预测二阶矩，equal 的矩为空；
- `media/`：沿用原生视频记录。

`avg_raw_weight_0`、`avg_raw_weight_1` 分别为实际 RGB/PCD 权重。融合权重已经归一化，不再对它们做额外 softmax。三个命令不读取示范动作或误差标签、不拟合参数；校准 JSON 仅提供冻结规则与半径。

先比较等权、point、robust 三组总体及逐任务成功次数，保留所有结果，不通过这轮结果重新挑选 checkpoint 或调整半径。每任务 20 次的成功率分辨率为 5 个百分点，少量成功次数差异不能直接确认为稳定提升。这轮仿真也不等同于使用示范数据中预留的 16 条 test 轨迹。

```bash
python -m pytest tests/test_error_moment_fusion.py tests/test_error_moments.py tests/test_moment_calibration.py -q
```

24 项合成数据测试通过，新增检查覆盖并行历史通道隔离/清空、每次规划一次统计前向、两专家共享每个带噪动作、解析权重以及原采样循环输出一致性。未加载真实 checkpoint、未运行真实模型或仿真。

### 15.1 五组统一初始化与随机数的评测

原入口现在支持 `--fusion rgb|pcd|equal|point|robust` 五组。rgb/pcd 是纯专家基线：每个去噪步只执行指定专家的去噪网络，不调用统计模型或另一专家的去噪网络。五组使用相同环境传感器配置、动作归一化和扩散 scheduler；两个编码器可同时计算，但纯专家输出不依赖另一编码器。

修正原来的种子失效路径：初始化函数带种子 reset 后，runner 实际开测又无种子 reset，而环境曾无条件调用 `np.random.seed(None)`。现在并行 runner 实际开测显式传入该批每个环境的 seed；环境只有收到非空 seed 才重置 NumPy/Python 随机数。无种子 reset 保留已有随机流。模型初始化也统一设置 NumPy/Python/PyTorch seed，并关闭 cuDNN benchmark、开启 cuDNN deterministic；扩散仍使用独立 generator。

这能减少初始环境和采样随机性的差异，不保证物理引擎及图形渲染逐位一致。`eval_log.json` 的 `episode_results[seed].initial_observation` 保存初始各传感器观测 SHA256，以及关节/夹爪位置，可核对同一任务和 seed 在五组中的初始观测。观测哈希不同应进一步检查机器人状态及视觉差异；哈希也不是完整物理状态的序列化。

```bash
cd /home/dbdx/NEUZZ/policyconsensus
conda activate policy-consensus
export PARALLEL_RLBENCH=1
export QT_QPA_PLATFORM=xcb
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for mode in rgb pcd equal point robust; do
  xvfb-run -a python -u scripts/eval_policy.py sim \
    -c output/experts/rgb_seed42/checkpoints/latest.ckpt \
    --pcd-checkpoint output/experts/pcd_seed42/checkpoints/latest.ckpt \
    --calibration output/error_moments/calibration_shared_history4_seed42/calibration.json \
    --fusion "$mode" --eval-seed 42 --test-start-seed 1000 \
    -o "output/eval_matched5_seed42/$mode" -n 80 -d cuda:0 || break
done
```

每组 8 环境并行、4 任务各 20 次、8 视频；五组按顺序前台执行，共 400 次。五组都传入 RGB/PCD 路径用于来源对齐，`--fusion pcd` 仍是纯 PCD 输出。新目录不覆盖先前三组或旧专家结果。此协议修改了种子生效方式，旧结果不视为五组中的匹配基线，应按新命令重新测试。

相关合成测试现在共 28 项，新增纯 RGB/PCD 与原 sampler 一致性、显式种子复现、无种子调用保持随机流和初始观测审计；没有启动真实实验。

## 16. 固定权重网格搜索基线

按 [GPC 的预训练策略凸组合与测试时权重搜索思路](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f7fc38fdd95fd146a471791b93ff9f12-Abstract-Conference.html)，增加固定权重扫描基线。这是本项目选择的搜索协议，不声称完全复现其所有实验设定。

RGB 权重 w 在 `[0,1]` 上按 0.1 步长取值，PCD 权重为 `1-w`。0%~100% 十等分是 10 个区间、11 个权重点。每个候选在全部状态、任务、重规划和去噪步上共用同一个 w。w=0、0.5、1 分别对应 PCD、等权和 RGB；本实现端点只执行对应专家去噪网络，不调用统计模型。其他固定权重也不调用统计模型，融合仍在同一带噪动作上的 epsilon 预测层进行。

```bash
cd /home/dbdx/NEUZZ/policyconsensus
conda activate policy-consensus
export PARALLEL_RLBENCH=1
export QT_QPA_PLATFORM=xcb
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for weight in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
  xvfb-run -a python -u scripts/eval_policy.py sim \
    -c output/experts/rgb_seed42/checkpoints/latest.ckpt \
    --pcd-checkpoint output/experts/pcd_seed42/checkpoints/latest.ckpt \
    --calibration output/error_moments/calibration_shared_history4_seed42/calibration.json \
    --fusion fixed --rgb-weight "$weight" \
    --eval-seed 42 --test-start-seed 1000 \
    -o "output/eval_fixed_grid_seed42/rgb_$weight" \
    -n 80 -d cuda:0 || break
done

python scripts/eval_policy.py summarize-fixed -i output/eval_fixed_grid_seed42
```

每点 4 任务各 20 次，共 880 次仿真。沿用 8 环境并行、相同任务/环境种子/扩散随机种子及每点 8 条视频；各点单独进程前台执行，失败停止循环。传入校准文件用于绑定同一对专家和相同采样协议，固定融合不使用其半径或统计网络预测权重。

汇总入口只读取已有日志，不执行仿真。打印各权重总体/逐任务成功率和成功次数，输出 `fixed_weight_summary.csv`、`fixed_weight_summary.json`。汇总校验候选权重、专家哈希、种子、并行数、测试总数、任务分配以及成功次数与总体成功率的一致性；缺失候选明确列出，最佳权重保留并列结果。逐任务最佳固定权重可以另作任务内搜索基线，但不可与单一全局最佳固定权重混为同一个比较。

这批 80 个起始环境用于搜索，网格最大成功率是选优后的搜索集成绩。选定并冻结最佳全局固定 w 后，建议用新环境种子 2000~2079、扩散种子 43，重新比较该固定 w、point、robust（等权可另保留），不继续用最终结果调参。若条件权重优于单专家但低于最佳固定权重，支持相对单专家的组合收益，尚不足以证明条件估计优于固定权重搜索。若匹配最佳固定权重，可讨论无需部署时再扫描权重的实用性，同时报告统计模型的额外训练数据和成本。

方案 A 的权重随观测和重规划时刻变化，在一次规划的扩散步骤内固定；扩散步时变权重仍属于尚未实现的方案 B。

新增 11 个权重点的合成采样一致性，以及搜索汇总、缺失结果和协议不匹配检查；全部相关测试共 45 项通过。未执行真实网格搜索。

## 17. 方案冻结后的最终 16 条示范轨迹测试

用户确认方案已确定，现开放原 manifest 的 test 分区（每任务 4 条，共 16 条）。通过原采集器和原离线校准/报告 workspace 增加最终评估模式，没有新建重复评测脚本。

冻结协议保存于 `splits/final_error_moments_protocol.json`：固定 RGB 权重 0.6、现有两个 latest 专家、history=4 的 best_risk 统计模型、现有校准 JSON，以及原始/风险数据划分文件的路径和 SHA256；噪声种子 2026、规划间隔 8、每时间步 8 次 MC。test 使用独立的噪声角色编号 3，区别于训练、验证和校准。协议在读取真实 test 标签前写入；真实采集和评估仍由用户运行。

```bash
cd /home/dbdx/NEUZZ/policyconsensus
conda activate policy-consensus

# 1. 对最终 test 的 16 条轨迹构造配对误差
python -u scripts/manip_policy.py --config-name=collect_error_moments_test

# 2. 读取冻结模型和校准半径，执行最终离线比较
python -u scripts/manip_policy.py --config-name=evaluate_error_moments_test
```

第一步中断可在相同配置后加 `resume=true`。两步是示范轨迹上的离线操作，不启动 RLBench，不需要 xvfb-run；必须等第一步完成后再执行第二步。最终缓存路径为 `output/paired_errors/test_latest_seed2026`。

第二步比较 RGB、PCD、等权、固定权重 0.6、point、robust 六种方法。使用相同配对误差，按完整 episode 均衡汇总，报告总体/逐任务/逐轨迹/逐扩散步 epsilon MSE、二阶矩预测 MSE、条件权重范围和固定半径在独立 test 上的轨迹覆盖情况。覆盖对象仍是有限 MC 标签，不能解释为任务成功率或潜在真实条件矩的保证。不重新训练、不选 checkpoint、不调整固定权重、不拟合半径。

输出目录 `output/error_moments/final_test_shared_history4_seed42/`：

- `final_test_report.json`：六种方法的误差及全部冻结来源；
- `final_test_predictions.npz`：预测矩、参考矩、权重、误差分数及 episode/时间索引；
- `final_test_config.yaml`：本次执行配置。

已存在最终报告时拒绝静默覆盖。训练/验证/校准与 test 的隔离、完整 episode 覆盖、冻结来源文件哈希及标签协议均会检查。最终离线结果与独立环境种子的闭环仿真成功率分别报告，不能互相替代。

49 项相关合成测试通过，包括完整最终评估流程在小型合成统计模型及缓存上的执行、冻结文件保持不变、test 禁止拟合归一化统计、分区隔离及拒绝覆盖结果。没有读取真实 test 轨迹、执行真实专家或启动仿真。
