# 实验日志 · MuJoCo 六基线复现与统一评测（双 H20）

机器：2× NVIDIA H20 97871 MiB（驱动 535.161.08）、384 核 CPU、2.2TB 内存。
注意：两张卡与本容器内另一租户的 ~22 个常驻 GPU 进程共享（nvidia-smi 可见），
本实验的训练因此全部走 CPU 路径（小网络在 CPU 上更快且不受 GPU 争抢影响），
GPU 仅用于 doctor 自检与个别烟测。CUDA 进程统一 `CUDA_MPS_PIPE_DIRECTORY=/tmp/mjb-no-mps`
绕开宿主 MPS daemon。

## 1. 上游固定（upstreams.lock.json 为准）

| 上游 | SHA | 角色 |
|---|---|---|
| robustrllib | a665ddf1 | 阶段A RARL-PPO 官方 train.py/evaluate.py（v5）；阶段B vendored 双PPO |
| sparc | 53f1e2e9 | 阶段A RMA 官方入口；qr_sac.py/wind_ant.py 缺失→本地分支 mjb-reconstruction 2ae8e3e 重建 |
| AD-RRL | dbe9a302 | DR-PPO 重建依据（论文附录E表4区间+其v3 FullObs语义+XML资产） |
| quantal-adversarial-rl | aae57b7b | QARL 移植源（mushroom-rl 1.9.1 语义核对） |
| mbpo (jannerm) | ac694ff9 | MBPO 算法参考（与UB口径对照） |
| rmbpo-tmlr | d58283f1 | 作者评估器+发布权重复测；阶段B网格/协议锚点 |
| unstable_Baselines 框架 | 5556a32 | unstable_baselines.* 导入路径（model_based_rl@2c2c4ac 为其 submodule，恰好=任务书锁定SHA） |

sparc 全部公开提交（11 个）均为 README 更新，qr_sac.py/wind_ant.py 从未存在（已核实）。

## 2. 环境（三套，互相隔离）

- `envs/mjb`（py3.11.17, torch2.5.1+cu121, mujoco2.3.7, gymnasium0.29.2=robustness-gymnasium fork -e, gym0.26.2, numpy1.26.4, sb3 2.3.2, pyarrow16.1.0, mushroom-rl 1.9.1, unstable_baselines 0.0.1 -e）
- `envs/mjb5`（gymnasium1.4.0, sb3 2.9.2, torch2.15.0+cu132, robustrllib -e）
- `envs/sparc`（gymnasium1.2.0, torch2.7.1+cu126, sb3 2.7.0, tyro0.9.27）

## 3. 入口（全部可重复执行）

```bash
python -m mujoco_baselines doctor                 # 环境自检（ALL PASS）
python -m mujoco_baselines smoke --algo <list>    # 各方法烟测
python -m mujoco_baselines run --protocol shared --workers N --resume
python -m mujoco_baselines run --protocol source --workers N --resume
python -m mujoco_baselines report [full]
```

实际启动命令（nohup setsid 派生，机器重启后可原样重跑）：

```bash
# 阶段B·四族（RMA/QARL/RARL/DR，12 workers + 12 孤儿）
python -m mujoco_baselines run --protocol shared --only rma,qarl,rarl_ppo,dr_ppo --workers 12 --resume
# 阶段B·模型双族（MBPO/RMBPO，20 workers，CPU device）
python -m mujoco_baselines run --protocol shared --only mbpo,rmbpo --workers 20 --resume
# 阶段A·全部 lane（16 workers：rma(sparc env)/rarl(mjb5 env)/qarl/dr(mjb env) + 复测）
python -m mujoco_baselines run --protocol source --workers 16 --resume
# 阶段A·作者发布权重复测（6 个，直接驱动器）
python -m mujoco_baselines.phase_a.rmbpo_eval <mbpo|rmbpo> <task>
```

## 4. 阶段A·作者评估器复测（已完成部分）

作者评估器原样执行（rmbpo-tmlr checkout 内，`--num-eval-runs 30`，MPLBACKEND=Agg）：

| 配置 | 作者网格均值（5 种子） | 任务书参照（热图估读） |
|---|---:|---:|
| MBPO Hopper 90格 | 1478 | ≈1630 |
| RMBPO Hopper 90格 (η=4) | 1945 | ≈2100 |
| RMBPO Walker 80格 (η=0.5) | 2399（共同64格 2594） | ≈2850 |
| MBPO Cheetah / MBPO Walker / RMBPO Cheetah | 运行中 | — |

计分口径注明：作者 evaluate_v2 将 done 步奖励清零（末步不计分）；
阶段B统一评估器计入末步。

## 5. 烟测记录（全部 PASS）

- dr_ppo / rarl_ppo / qarl / rma：20k 步 Hopper + 2k 步另两任务 + 2 格点评估 + 计数核对
- mbpo：6k 步全任务（学习曲线正常，holdout MSE 0.0066，train return 210+）
- rmbpo：bypass 回归测试两轮过（同输入同种子下与 MBPO 名义 rollout 位级一致；
  旁路关闭时 11000/11000 转移元素不同=辅助路径真实生效）
- 修正一处 RMBPO 辅助损失缺陷：鲁棒性项须用 名义diff+obs 重构绝对状态喂 critic
  与终止函数（原实现直接把 diff 当状态），已修复并重跑 bypass 测试

## 6. 已知规模与吞吐（实测）

- 阶段B：90 作业×3M 转移；RMA ~68 sps/作业（CPU）、QARL ~59、MBPO ~36（CPU）
- 阶段A：66 作业；作者复测每配置 12-16 分钟
- 故障重试：runner 对 rc≠0 作业自动重排队（≤2 次），支持从检查点续跑

## 7. 资源实况与调度（运行期发现）

- **cgroup CPU 配额 = 94 核**（/sys/fs/cgroup cpu.cfs_quota_us=9,400,000/100,000），
  尽管机器名义 384 核。初期 90 作业×8 线程造成 load 600 的线程风暴，已统一
  OMP_NUM_THREADS=2。
- **两卡被外部租户 ~22 个 GPU 进程长期占用**（nvidia-smi 可见，非本容器），
  全部训练走 CPU 路径（实测小网络 CPU 快 3-5 倍且不受争抢）。
- **带宽竞争**：并发作业越多单步越慢（70 作业 27ms/步 → 32 作业 ~12ms/步），
  watchdog（tools/watchdog.py）以 32 活跃作业为目标自动分期：P0 非模型族 →
  P1 模型双族 → P2 阶段A 训练 lane；SIGSTOP/CONT 树级暂停恢复。
- RMBPO 两处修复：鲁棒性损失的 pred_state 须 diff+obs 重构；辅助轮加
  max_train_iterations=2000 上限（论文"while improving"规则在拉长预算下不终止）。
  名义模型训练同样加上游计算包络上限 12000 iters/轮。
