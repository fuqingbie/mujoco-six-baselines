# GPU 节点四机交接包（round: gpu-release-20261119-01）

GPU 节点（原双 H20，本机）→ any3 / any7 / any8 三台 CPU 节点的源码与可恢复快照交接。
**SSH/代理 CONNECT 均不通（出口受限）**——交接内容已全部上传 GitHub Release（匿名可下载）：

**https://github.com/fuqingbie/mujoco-six-baselines/releases/tag/handoff-20261119-01**

| 资产 | 内容 | 体积 | SHA-256（前 16 位） |
|---|---|---|---|
| `source-package.tar.gz` | 完整源码（package_sha256=5b3208fa…800c342） | 0.5MB | — |
| `handoff-any3.tar.gz` | any3 名下 16 作业快照 | 88.8MB | 5eab2b81602eaeea |
| `handoff-any7.tar.gz` | any7 名下 15 作业快照 | 1883MB | d0eb1a5c3c219d89 |
| `handoff-any8.tar.gz` | any8 名下 7 作业快照 | 704MB | 7f32dc9c9b970326 |
| `*.sha256` | 各资产校验和 | — | — |

CPU 侧：匿名 `curl -L -O` 下载各资产 → `sha256sum -c` 校验 → 解包到工作根目录（得到 `checkpoints/<algo>/<task>/seed<N>/`）→ 源码指纹对齐 `staged-package.sha256` → 收据在仓库 `handoff/inbox/handoff.json`（approved: true，按 `target` 认领）。逐 ckpt 文件 SHA-256 也在收据内。
本地同等备份保留在 GPU 节点 `/root/mujoco-six-fleet-handoff/`。

## 目录内容

```
package/
  source-package.tar.gz     # 完整实验源码（mujoco_baselines/ + configs/frozen.json + upstreams.lock.json）
  staged-package.sha256     # 5b3208fabf343581935a7969f5d62aea249627d2e0e843178995b6f1d800c342
checkpoints/<algo>/<task>/seed<N>/
  config.json               # 该作业的冻结配置（含 train_seed）
  counters.json             # 释放时刻的日志计数（可能高于检查点步数，仅作核对）
  progress.jsonl            # 训练曲线
  ckpt/…                    # 可恢复状态（SHA-256 见 inbox/handoff.json）
  manifest.json             # 38 个作业的快照清单
inbox/
  handoff.json              # 最终收据（approved: true；GPU 旧队列已释放后签发）
```

## CPU 侧安装步骤（每台节点）

1. 工作根目录 `/data/mujoco-six-fleet`。
2. **对齐源码**：解包 `source-package.tar.gz`，用收据里的 `package_sha256` 校验（校验脚本即任务书第 3 节的指纹脚本）；把指纹写入 `/data/mujoco-six-fleet/staged-package.sha256`。若节点已有旧 staged 指纹（来自公开仓库缺 envs 文件的版本），必须用本包覆盖对齐，否则自动接收器拒收。
3. **检查点就位**：把 `checkpoints/<algo>/<task>/seed<N>/` 整目录上传到 `/data/mujoco-six-fleet/checkpoints/<algo>/<task>/seed<N>/`。state 文件 SHA-256 以 `handoff.json` 内 `checkpoint_sha256` 为准。
4. **收据安装**：`handoff.json` 先落 `inbox/handoff.json.tmp` 再 `mv` 原子改名（本包已是原子写好的成品）。三台共用同一份收据，各自只认领 `target` 为本机的条目。
5. **恢复验证**：从 `checkpoint_step` 起算，不是 counters.json 的日志步数（模型族 state_5000.pt=5000，即使日志到了 121k）。发现恢复到 0 或新预填充即标记恢复失败并回报。
6. 环境依赖见仓库 README（`bash tools/install_envs.sh` 建三套 conda env；CPU 节点只需 `mjb` 一套）。

## 作业分配总览（38 交接 + 22 GPU 保留 = 60 未完成 shared）

| 节点 | 作业 |
|---|---|
| any3 (16) | MBPO walker2d s0-3（ckpt 5000）、RMBPO hopper s1,2 + halfcheetah s0,1（ckpt 5000）、QARL halfcheetah s0-4（pending 全新）+ walker2d s0-2（ckpt ~6.5k） |
| any7 (15) | RMA hopper s0-4 + halfcheetah s0-4 + walker2d s0,1（ckpt 800k/1000k）、QARL hopper s0-2（ckpt 615-716k） |
| any8 (7) | RMA walker2d s2-4（ckpt 800k/1000k）、QARL hopper s3,4（ckpt 700/742k）+ walker2d s3,4（ckpt ~6.3k） |
| GPU 保留 (22) | MBPO hopper s0-4 + halfcheetah s0-4 + walker2d s4；RMBPO hopper s0,3,4 + halfcheetah s2,3,4 + walker2d s0-4（在训中，保活器看护） |

已完成（30，GPU 侧已验收）：DR-PPO 15/15、RARL-PPO 15/15（含全套评测落盘）。
阶段 A：作者复测 6/6 完成；60 个 source 训练作业归 GPU（在 shared 之后）。

## 多机注意

- `save_records` 对同一 `episodes/shared/<algo>/<task>/episodes.parquet` 是读改写——**多机禁止并发写同一文件**。各节点先按 seed 隔离存放，最终汇总按 `(protocol,algo,task,seed)` 去重、校验每格 30 回合后再并表。
- 收据 `notes` 字段记录了释放口径与 GPU 侧保留清单。
