# 基于MMPC的长期价值驱动视触策略共识与冲突感知组合技术路线

> 本文档以MMPC项目、MMPC数据和MMPC评测任务为唯一项目基线。本文不使用UniVTAC，不研究模态内子策略的语义分解，也不把新增模态能力作为论文贡献。研究只聚焦于MMPC策略级视触组合中的两个问题：权重依据与冲突处理。

## 1. 研究问题与边界

### 1.1 核心问题

MMPC通过Router直接产生视觉和触觉策略的组合权重：

\[
\epsilon_w=w\epsilon_V+(1-w)\epsilon_T,
\qquad w\in[0,1].
\]

Router权重能够随当前观测变化，但权重主要由去噪训练目标间接获得，难以说明：

1. 为什么当前应该使用这个权重；
2. 该权重对后续任务成功有什么影响；
3. 什么时候保持现有权重，什么时候切换；
4. 当任何标量权重都不能形成有效动作时，应该如何处理专家冲突。

本文的两个研究模块为：

1. **长期价值驱动的动态共识权重优化**：把权重视为高层时序控制变量，学习不同候选权重对未来闭环任务结果的影响，再依据风险调整后的长期价值选择和切换权重。
2. **标量加权失效时的冲突感知组合**：当最佳标量权重仍不足时，在视觉主导、触觉主导等有界组合方式中选择更高价值的共识算子。

### 1.2 明确不做的内容

- 不使用UniVTAC作为数据、训练或实验平台；
- 不强调Context/Local、Coarse/Fine等模态内子策略语义；
- 不声称当前MMPC代码已经形成了上述子策略专长；
- 不把增加新模态或即插即用模态扩展作为实验贡献；
- 不建立V、T、VT三个互相独立的联盟；
- 不在首版中训练独立的事件识别网络；
- 不将GelSight Mini替换原触觉设备本身作为方法创新。

### 1.3 视觉和触觉策略接口

本文只要求MMPC提供两个模态级输出：

\[
\epsilon_V(A_k,k,o_t^V,q_t),
\qquad
\epsilon_T(A_k,k,o_t^T,q_t).
\]

MMPC内部是否包含一个或多个UNet、是否平均、是否共享编码器，都由发布代码和选定配置决定。本文通过适配层取得最终视觉Score和触觉Score，不研究其内部子策略语义。

为避免架构差异造成不公平，所有对照方法必须使用同一视觉和触觉专家checkpoint。本文只替换或扩展权重决策和共识组合部分。

## 2. 项目基线与实验平台

### 2.1 代码与数据

项目代码：MMPC发布仓库 `policyconsensus`。

主要基线文件：

- `modular_policy/policy/dp_unets_spec.py`；
- MMPC发布的RGB+Tactile配置；
- MMPC提供的真实世界数据、训练脚本和评测接口；
- 需要仿真分支时，使用MMPC项目已有仿真任务和环境接口扩展状态保存/恢复功能。

### 2.2 两类MMPC基线

为了兼顾论文复现和方法公平，保留两类基线：

1. **Released MMPC**：不改变发布代码的权重网络和组合方式，作为原论文复现基线。
2. **Controlled MMPC**：使用与本文完全相同的视觉/触觉专家接口、历史长度和动作执行频率，仅使用普通Router直接输出权重，作为主要公平对照。

如果发布代码内部存在多个同模态模块，适配层只需要给出该配置最终用于组合的视觉和触觉模态Score。本文不为这些模块附加新的语义标签。

### 2.3 不验证新增模态能力

论文只研究视觉和触觉两个模态。实验不需要加入音频、深度或额外传感器，也不需要证明新增模态时无需重训。MMPC关于组合灵活性的主张属于基线背景，不属于本文验证范围。

## 3. 权重的分布含义

设视觉和触觉策略分别对应：

\[
p_V(A\mid h_t),\qquad p_T(A\mid h_t),
\]

其中 \(h_t\) 为部署时可见历史。定义对数线性组合：

\[
p_w(A\mid h_t)
=\frac{1}{Z(h_t,w)}
p_V(A\mid h_t)^w
p_T(A\mid h_t)^{1-w}.
\]

当 \(w\) 只依赖物理历史 \(h_t\)，并在一次动作块的完整扩散采样中保持不变时：

\[
\nabla_A\log p_w
=w\nabla_A\log p_V+(1-w)\nabla_A\log p_T.
\]

因此：

\[
\epsilon_w=w\epsilon_V+(1-w)\epsilon_T.
\]

这里的权重是对数线性组合中视觉与触觉策略的相对指数参数，而不仅是Router的注意力分数。

理论边界：专家Score是近似量；两模态未必条件独立；若权重在去噪过程中依赖当前噪声动作并不断变化，则上述固定联合分布解释不再直接成立。因此首版在一次动作块内固定权重。

## 4. 权重Oracle预实验

### 4.1 Oracle的目的

Oracle不是可部署算法，而是回答一个先决问题：

> 在MMPC任务和专家不变时，不同闭环状态是否真的需要不同权重，且正确选择权重是否能明显超过最佳固定权重和MMPC Router？

如果Oracle都没有提升空间，则没有必要训练长期价值模型。

### 4.2 决策状态来源

先运行以下策略采集闭环轨迹：

- Released MMPC；
- 最佳固定权重；
- 视觉边界 \(w=1\)；
- 触觉边界 \(w=0\)。

在每次MMPC重新调用 `predict_action()` 的重规划时刻保存候选分支状态。采样应覆盖：

- 正常成功轨迹；
- 失败和恢复轨迹；
- 接触前、接触建立、稳定接触和局部修正状态；
- 视觉遮挡、触觉弱响应和专家输出分歧状态。

阶段标签只用于分层统计，不作为价值模型必须输入的真值。

### 4.3 状态恢复验收

在MMPC仿真环境中扩展状态保存和恢复，尽量保存：

- 机器人关节位置与速度；
- 末端和夹爪状态；
- 物体位姿与速度；
- 环境随机数状态；
- 观测历史缓存；
- 扩散采样随机数；
- 环境可访问时的接触或动力学内部状态。

先从同一状态重复执行相同权重分支，计算回报方差。只有不同权重的效应明显大于恢复噪声，才称为配对状态分支。

如果MMPC环境不能可靠恢复接触状态，则改用：

1. 相同初始配置和随机种子；
2. 到达匹配决策时刻后随机分配候选权重；
3. 大量重复实验估计条件平均效果。

此时不声称得到逐状态精确反事实Oracle。

### 4.4 候选权重与分支执行

首版候选集合：

\[
\mathcal W=\{0,0.25,0.5,0.75,1\}.
\]

对同一个决策状态 \(s_t\)，每个权重分支：

1. 使用相同观测历史；
2. 使用相同扩散初始噪声；
3. 完成一次动作块采样；
4. 执行相同长度的动作前缀；
5. 之后切换到同一个固定continuation policy；
6. 运行到固定评估时域或任务终止；
7. 对多个环境种子或扰动重复。

continuation policy首版使用冻结的Released MMPC或Controlled MMPC。所有权重分支必须使用同一个continuation policy。

### 4.5 三类Oracle

#### Chunk Oracle

只比较当前动作块或固定短时域回报：

\[
w_{oracle}^{K}(s_t)
=\arg\max_{w\in\mathcal W}G_{t:t+K}(s_t,w).
\]

它用于验证权重的即时可区分性，不能称为全任务最优权重。

#### One-step-deviation Oracle

当前权重执行一个动作前缀，随后所有分支使用相同continuation policy，比较最终成功或长期回报：

\[
w_{oracle}^{dev}(s_t)
=\arg\max_w
G(s_t,w;\pi_{cont}).
\]

它测量相对于固定后续策略的一次权重干预价值，是首版最重要的Oracle。

#### Privileged Dynamic Oracle

在线仿真评测时，每个重规划点都临时克隆环境，对所有候选权重展开分支，再在真实仿真回报下选择最优分支。该方法使用部署不可用的仿真特权，只作为性能上界，计算成本高，不用于训练和真机部署。

### 4.6 Oracle继续条件

至少满足：

1. 不同状态的最佳权重具有稳定差异；
2. One-step-deviation Oracle明显优于最佳固定权重；
3. Privileged Dynamic Oracle或近似动态上界优于MMPC Router；
4. 该优势不仅来自人为关闭传感器；
5. 效应大于状态恢复和测试trial噪声。

否则停止长期价值路线或缩小论文主张。

## 5. 历史状态定义

### 5.1 为什么不能只用当前输入

相同单帧视觉和触觉可能对应不同物理历史，例如刚建立接触、正在失去接触或滑移后短暂稳定。因此使用因果历史，而不把它称为完整真实信念状态。

更准确的名称为：

> **部署可见历史状态**，deployment-observable history state。

记为：

\[
h_t=H_\omega(z^V_{t-L+1:t},z^T_{t-L+1:t},
q_{t-L+1:t},a_{t-L:t-1},w_{t-L:t-1}).
\]

其中：

- \(z^V\)：MMPC视觉编码器输出；
- \(z^T\)：MMPC触觉编码器输出；
- \(q\)：机器人本体状态；
- \(a\)：实际执行动作，而不是仅使用预测动作；
- \(w\)：此前真正执行的共识权重。

可选加入：传感器时间戳、数据缺失mask、当前权重保持步数。不得加入部署时不可访问的物体真实位姿、接触求解器状态或未来观测。

### 5.2 编码器

首版采用GRU：

\[
h_t=\operatorname{GRU}(x_t,h_{t-1}).
\]

建议初始历史长度为最近8个策略重规划时刻，再通过 \(L\in\{1,4,8,16\}\) 消融确定。历史以策略调用时刻为单位，不直接混用摄像机原始帧率。

### 5.3 必要对照

- 当前帧MMPC Router；
- 当前帧价值模型；
- Recurrent Router；
- 历史价值模型；
- 历史顺序打乱；
- 历史截断；
- 仅权重平滑；
- 显式接触阶段规则。

只有历史价值模型在控制容量和数据预算后仍有提升，才能说明历史用于效用判断，而非简单识别时间进度。

## 6. 长期价值定义

### 6.1 权重作为高层控制动作

一次高层决策选择权重 \(w_t\)，低层MMPC专家据此生成一个动作块并执行前 \(K\) 步。权重价值定义为：

\[
Q^\pi(h_t,w_t)
=\mathbb E_\pi
\left[
\sum_{j=0}^{T-t}\gamma^jr_{t+j}
\mid h_t,w_t
\right].
\]

当后续继续进行最优权重选择时：

\[
Q^*(h_t,w)
=\mathbb E\left[
R_{t:t+\tau}
+\gamma^\tau
\max_{w'}Q^*(h_{t+\tau},w')
\right].
\]

这一定义来自把权重作为高层动作后的Bellman最优方程。实际系统只近似该价值，不声称获得真实全局最优控制。

### 6.2 主价值和辅助价值

主价值优先使用剩余任务成功概率：

\[
Q_{succ}(h_t,w)
=P(\text{任务在剩余时域成功}\mid h_t,w).
\]

辅助头预测：

- 固定时域任务进展；
- 距离终止或失败的时间；
- 滑移、卡死或碰撞概率；
- 任务定义允许时的几何误差。

权重选择采用词典序：先排除违反硬约束或失败概率过高的候选，再最大化成功概率，价值接近时选择切换更少或耗时更低的候选。避免用大量任意reward系数混成唯一标量。

## 7. 长期价值的两阶段训练

### 7.1 阶段一：固定continuation的监督价值预训练

使用Oracle预实验产生的数据：

\[
\mathcal D_{branch}
=\{h_t,w,G_{t:t+K},G_{terminal},y_{succ},h_{t+\tau}\}.
\]

首先学习两个可直接监督的目标。

有限时域价值：

\[
Q_K(h_t,w)
\approx
\mathbb E[G_{t:t+K}\mid h_t,w].
\]

固定continuation成功价值：

\[
Q_{dev}(h_t,w)
\approx
P(\mathrm{Success}\mid h_t,w,\pi_{cont}).
\]

损失：

\[
\mathcal L_{stage1}
=\mathcal L_{Huber}(\hat Q_K,G_K)
+\lambda_s\operatorname{BCE}(\hat P_{succ},y_{succ})
+\lambda_r\mathcal L_{rank}.
\]

阶段一回答：部署可见历史能否预测“当前采用不同权重会造成什么后果”。它还不是完整的最优长期价值。

### 7.2 阶段二：Bellman价值迭代

收集高层转移：

\[
(h_t,w_t,R_{t:t+\tau},h_{t+\tau},done).
\]

使用阶段一模型初始化，构造目标：

\[
y_t=
\begin{cases}
R_{t:t+\tau},&done,\\
R_{t:t+\tau}+\gamma^\tau
\max_{w'\in\mathcal W}
\bar Q(h_{t+\tau},w'),&\text{otherwise}.
\end{cases}
\]

训练：

\[
\mathcal L_{stage2}
=\mathcal L_{Huber}(Q_\phi(h_t,w_t),y_t)
+\lambda_r\mathcal L_{rank}
+\lambda_c\mathcal L_{calibration}.
\]

实现要求：

- 使用慢更新target network \(\bar Q\)；
- 可使用双价值网络的较小值抑制过估计；
- 候选权重保持离散，避免离线数据未覆盖的连续权重外推；
- 数据按轨迹、物体和场景划分；
- 所有候选权重具有基本覆盖；
- 训练若使用额外分支数据，Router强化基线获得相同交互预算。

### 7.3 闭环再标注

初始价值模型上线后会访问新的状态分布。执行有限轮：

1. 使用当前价值策略采集闭环状态；
2. 在仿真中对这些状态补充分支权重标签；
3. 更新价值模型；
4. 在独立验证集重新校准；
5. 保持所有方法相同的总交互预算。

最多进行预先设定的少量迭代，避免把无限在线强化学习包装成简单价值选择。

## 8. 权重选择与切换

### 8.1 与MMPC一致的周期重评估

MMPC当前代码在一次 `predict_action()` 中计算一次权重，并在整个扩散采样中使用同一权重。执行动作前缀后，下一次调用策略时根据新观测重新计算。

本文保持同样的基础频率：

1. 新观测到达；
2. 更新历史状态；
3. 评估候选权重；
4. 在完整扩散采样中固定选定权重；
5. 执行动作前缀；
6. 下一次重规划时重新评估。

### 8.2 风险调整选择

将上一权重和保持时间作为历史的一部分，定义：

\[
U_t(w)=
\hat Q(h_t,w)
-\beta\hat\sigma(h_t,w)
-\lambda_{sw}(w-w_{t-1})^2.
\]

候选权重：

\[
w_t^{cand}=\arg\max_{w\in\mathcal W}U_t(w).
\]

只有当：

\[
U_t(w_t^{cand})-U_t(w_{t-1})>\delta
\]

且当前权重已保持至少 \(d_{min}\) 个重规划周期时才切换。否则继续使用 \(w_{t-1}\)。

\(\beta,\lambda_{sw},\delta,d_{min}\) 只在验证集确定。

## 9. 事件触发重评估

### 9.1 首版定位

事件触发不是主方法的必要部分。主版本完全可以只采用与MMPC相同的周期性动作块重规划。这样实现简单、比较公平，也不需要另建事件识别网络。

事件触发只作为可选工程增强和消融：当某些突发变化发生在动作块执行中间时，提前终止剩余动作前缀并重新评估权重。

### 9.2 事件只触发重评估

事件不直接决定新权重。例如检测到接触后不能硬编码 \(w=0.25\)。正确流程为：

\[
\text{事件发生}
\rightarrow
\text{提前重规划}
\rightarrow
\text{重新计算所有候选权重价值}
\rightarrow
\text{按同一切换准则决策}.
\]

### 9.3 无学习事件定义

首版只使用MMPC项目和机器人接口中能够直接计算的信号，并采用双阈值和连续帧确认。

#### 接触建立/丢失

定义触觉活动量：

\[
c_t=\frac{1}{N}\sum_{i=1}^{N}|x_{t,i}-x_{0,i}|,
\]

其中 \(x_0\) 是无接触基线。若连续 \(M\) 帧：

\[
c_t>\tau_{on}
\]

则判定接触建立；若连续 \(M\) 帧：

\[
c_t<\tau_{off},\qquad \tau_{off}<\tau_{on},
\]

则判定接触丢失。

#### 触觉突变或疑似滑移

若触觉输入支持时序差分，定义：

\[
d_t^T=\|z_t^T-z_{t-1}^T\|.
\]

当 \(d_t^T\) 超过正常接触条件下验证集分位数阈值，并持续规定帧数时触发重评估。除非有真实滑移标签或可靠标定，只称为“触觉突变事件”，不直接称为滑移检测。

#### 传感器异常

- 时间戳长时间不更新；
- 图像或触觉帧缺失；
- 数值饱和；
- 输入超出训练分布的简单统计阈值。

#### 执行偏差

根据命令动作与机器人实际状态变化：

\[
e_t^{exec}=\|\Delta q_t^{cmd}-\Delta q_t^{obs}\|.
\]

连续超阈值时提前重规划。

### 9.4 事件阈值确定

- 只用训练集和验证集设置；
- 使用迟滞阈值避免反复触发；
- 使用连续 \(M\) 帧确认；
- 报告事件误触发率、漏触发率和触发延迟；
- 如果事件触发相对固定周期没有明显收益，从主方法中删除。

## 10. 冲突感知非线性组合

### 10.1 正确的标量权重失效定义

因为 \(w=0\) 和 \(w=1\) 已包含在标量权重集合中，所以：

\[
\max_{w\in[0,1]}Q(h,w)
\geq
\max(Q(h,0),Q(h,1)).
\]

不能使用“最佳线性权重低于最佳边界权重”作为失效条件。

内部线性协同增益定义为：

\[
G_{linear}(h)
=\max_{0<w<1}Q(h,w)
-\max(Q(h,0),Q(h,1)).
\]

当 \(G_{linear}\leq0\) 时，只能说明内部标量融合没有优于最佳边界策略。

非线性共识的真实增益定义为：

\[
G_{nonlinear}(h)
=\max_{g\in\mathcal G_{nonlinear}}Q(h,g)
-\max_{w\in[0,1]}Q(h,w).
\]

只有 \(G_{nonlinear}\) 在多状态、多任务中稳定为正，冲突机制才构成第二项贡献。

### 10.2 候选非线性算子

视觉主导、触觉有界修正：

\[
\epsilon_{V-dom}
=\epsilon_V+\lambda P_T(\epsilon_T-\epsilon_V).
\]

触觉主导、视觉有界修正：

\[
\epsilon_{T-dom}
=\epsilon_T+\mu P_V(\epsilon_V-\epsilon_T).
\]

首版使用具有明确动作坐标含义的维度掩码和幅值限制构造 \(P_T/P_V\)。没有物理标定时不称为真实接触子空间。

统一配置：

\[
c=(g,\theta_g),
\]

其中线性模式参数为 \(w\)，视觉主导参数为 \(\lambda\)，触觉主导参数为 \(\mu\)。使用相同价值框架比较候选配置：

\[
c_t^*=\arg\max_c
[\hat Q(h_t,c)-\beta\hat\sigma(h_t,c)
-\operatorname{SwitchCost}(c,c_{t-1})].
\]

## 11. 完整训练和在线流程

### 11.1 离线/仿真训练

```text
MMPC项目与数据
    ↓
复现Released MMPC和Controlled MMPC
    ↓
记录每次动作块重规划状态
    ↓
MMPC仿真环境状态恢复验收
    ↓
候选权重配对分支
    ↓
Chunk Oracle与One-step-deviation Oracle
    ↓
阶段一：有限时域/固定continuation价值预训练
    ↓
阶段二：Bellman价值迭代
    ↓
闭环再标注与验证集校准
    ↓
切换代价、滞回与最短保持时间
    ↓
必要时增加非线性共识候选
```

### 11.2 在线部署

```text
读取MMPC视觉、触觉与机器人状态
    ↓
更新部署可见历史状态 h_t
    ↓
评估候选权重的价值与不确定性
    ↓
减去权重切换代价
    ↓
判断是否保持或切换
    ↓
在本次完整扩散采样中固定权重/共识模式
    ↓
生成动作块并执行前K步
    ↓
下一周期重新评估
    ↓
可选：接触/触觉突变/传感异常时提前重规划
```

## 12. 代码实现路线

保留原文件作为Released MMPC基线，新建：

```text
modular_policy/policy/value_guided_consensus.py
modular_policy/model/consensus/mmpc_expert_adapter.py
modular_policy/model/consensus/history_encoder.py
modular_policy/model/consensus/weight_value.py
modular_policy/model/consensus/value_uncertainty.py
modular_policy/model/consensus/switch_controller.py
modular_policy/model/consensus/event_monitor.py        # 可选
modular_policy/model/consensus/nonlinear_operator.py   # 第二阶段
modular_policy/workspace/collect_weight_branches.py
modular_policy/workspace/train_weight_value_stage1.py
modular_policy/workspace/train_weight_value_stage2.py
modular_policy/workspace/eval_value_consensus.py
```

统一接口：

```python
adapter.predict_visual_score(noisy_action, step, obs) -> score_v
adapter.predict_tactile_score(noisy_action, step, obs) -> score_t
history_encoder.update(obs, executed_action, executed_weight) -> history
value_model(history, candidate_weight, prev_weight) -> value, uncertainty
switch_controller.select(values, uncertainties, prev_weight, dwell) -> weight
consensus.compose(score_v, score_t, weight, mode) -> score
```

Released MMPC的checkpoint、专家参数和normalizer保持冻结。阶段一不反向更新专家。

## 13. 实验基线与消融

### 13.1 必要基线

1. Vision-only，\(w=1\)；
2. Tactile-only，\(w=0\)；
3. MMPC发布实现；
4. Controlled MMPC当前状态Router；
5. Recurrent Router；
6. 固定 \(w=0.5\)；
7. 验证集最佳固定权重；
8. 推理期权重搜索的公平基线；
9. 阶段规则权重；
10. 阶段一短时价值选择；
11. 阶段二长期价值选择；
12. 长期价值 + 切换机制；
13. 完整方法 + 非线性共识；
14. Chunk/Deviation/Dynamic Oracle上界。

### 13.2 历史与价值消融

- 当前输入与历史状态；
- 有/无动作历史；
- 有/无以前权重；
- 历史长度1/4/8/16；
- 历史打乱；
- 固定continuation价值与Bellman价值；
- 仅回归与回归+排序；
- 有/无价值不确定性；
- 不同候选权重网格；
- 有/无闭环再标注。

### 13.3 切换消融

- 每个动作块直接取最大值；
- 权重指数平滑；
- 切换代价；
- 切换代价 + 优势阈值；
- 加最短保持时间；
- 周期重评估；
- 周期 + 无学习事件触发。

### 13.4 公平性控制

- 强Router基线使用相同历史编码器容量；
- 直接回报优化Router获得相同仿真交互预算；
- 所有方法使用同一专家checkpoint；
- 所有方法使用相同动作执行频率；
- 额外分支数据量、训练GPU时间和部署延迟全部报告。

## 14. 评价指标

### 14.1 任务指标

- 任务成功率；
- 完成时间；
- 接触失败、滑移代理和卡死次数；
- 回退率；
- 动作平滑性；
- 正常场景与扰动场景性能。

### 14.2 权重价值指标

- 有限时域价值MAE；
- 成功概率Brier score和可靠性图；
- Spearman/Kendall权重排序相关；
- top-1候选权重准确率；
- 相对Oracle的权重选择regret；
- 不确定性校准；
- 不同任务阶段的权重价值曲线。

### 14.3 切换指标

- 每回合切换次数；
- 无效切换率；
- 切换前后真实回报差；
- 接触变化后的响应延迟；
- 因切换造成的动作抖动；
- 事件误触发、漏触发和触发延迟。

### 14.4 冲突指标

- \(G_{linear}\) 分布；
- \(G_{nonlinear}\) 分布；
- 非线性模式选择regret；
- 正常互补状态的性能损失；
- 高分歧状态的恢复率。

## 15. 里程碑和停止条件

| 阶段 | 交付物 | 继续条件 |
|---|---|---|
| M0 MMPC复现 | 发布基线、权重日志、任务结果 | 基线结果和接口稳定 |
| M1 分支验收 | 同权重重复方差与多权重收益表 | 权重效应大于恢复噪声 |
| M2 Oracle预实验 | Chunk与Deviation Oracle | 动态选择优于最佳固定权重 |
| M3 历史价值阶段一 | 排序、校准和短时regret | 部署可见历史能预测权重效果 |
| M4 Bellman阶段二 | 长期价值与闭环结果 | 优于同预算Recurrent Router |
| M5 切换机制 | 周期、滞回和保持时间消融 | 减少无效切换且不降低成功率 |
| M6 冲突共识 | 非线性与最佳标量比较 | \(G_{nonlinear}\) 稳定为正 |
| M7 真机验证 | UR5e + GelSight Mini结果 | 优势在真实可见历史下保持 |

停止或转向条件：

- Oracle相对最佳固定权重没有明显增益；
- Released或Recurrent MMPC已经接近Oracle；
- MMPC环境无法形成可信分支，群体随机实验成本也不可接受；
- 历史价值依赖部署不可见状态；
- 长期价值没有优于短时规则或普通Router；
- 切换机制主要通过不动或回退获得低风险；
- 非线性共识没有超过最佳标量权重；
- 方法优势完全来自更多交互数据。

## 16. 论文主张与答辩口径

### 16.1 推荐题目

**Long-Horizon Value-Guided Visuotactile Policy Consensus with Conflict-Aware Composition**

中文：

**长期价值驱动的视触策略共识与冲突感知组合**

### 16.2 核心主张

1. 在MMPC视触策略共识中，将组合权重从Router的直接输出重新定义为具有对数线性分布含义的高层时序控制变量；
2. 使用MMPC环境内的配对权重分支，学习部署可见历史条件下的长期权重价值；
3. 通过风险调整价值、切换代价和滞回决定权重保持与切换；
4. 仅在标量权重族不足时，使用有界非线性共识处理专家冲突；
5. 通过Oracle、同预算Recurrent Router、价值校准和regret评估验证方法，而不是只展示权重曲线。

### 16.3 “为什么这样定义价值”的回答

> 权重决定未来一个动作块所使用的联合策略分布，因此它是高层控制动作。长期价值定义为当前部署可见历史下，采用该权重执行一个控制段并随后继续最优权重控制时的剩余任务成功概率或长期回报，该定义来自Bellman最优方程。阶段一先学习固定continuation下可直接监督的分支价值，阶段二再通过高层转移和target network进行Bellman价值迭代。

### 16.4 “Oracle怎么做”的回答

> 在MMPC仿真环境的策略重规划时刻保存可恢复状态，对候选权重使用相同历史、扩散噪声、动作前缀和后续策略进行配对分支。短时Oracle比较固定窗口收益，Deviation Oracle比较执行一个权重前缀后在共同continuation policy下的最终结果，仿真动态Oracle则在每个重规划点使用环境克隆展开所有权重，只作为特权上界。状态不能可靠恢复时，改用匹配初始条件下的随机权重分配实验。

### 16.5 “什么时候切换”的回答

> 主版本与MMPC一致，在每次动作块重规划时重新评估权重，并在一次扩散采样中保持权重不变。只有新权重的风险调整长期价值相对当前权重超过切换代价和滞回阈值，并满足最短保持时间时才切换。事件触发不是核心模块；若使用，只由接触活动量、触觉突变、传感器异常和执行偏差等直接信号提前触发重规划，不直接指定新权重。

## 17. 最小可行版本

第一版只实现：

1. MMPC发布代码和数据复现；
2. 从现有MMPC结构取得视觉Score和触觉Score的适配层；
3. 候选权重 \(\{0,0.25,0.5,0.75,1\}\)；
4. MMPC仿真环境的状态恢复验收；
5. Chunk Oracle和One-step-deviation Oracle；
6. 8步历史GRU；
7. 固定continuation价值预训练；
8. 离散权重Bellman价值迭代；
9. 每动作块周期重评估、切换代价、滞回和最短保持时间；
10. Released MMPC、Recurrent Router、最佳固定权重和Oracle完整对照。

事件触发和非线性冲突共识只有在最小版本证明核心问题存在后再实现。
