# BC training loop + GGL warm-start

This document covers (1) the behavior-cloning training loop, (2) GGL's on-disk checkpoint format, (3) warm-start strategies, (4) post-warm-start training, and (5) a concrete recipe for getting a BC-trained actor into a GGL checkpoint folder that auto-loads.

Primary GGL references (paths are inside `/home/seis/code/GigaLearnCPP-Leak/`):

- `GigaLearnCPP/src/private/GigaLearnCPP/Util/Models.h` / `Models.cpp` — Model class + `Save`/`Load`
- `GigaLearnCPP/src/public/GigaLearnCPP/Util/ModelConfig.h` — `PartialModelConfig` (layer sizes, activation, layer-norm flag)
- `GigaLearnCPP/src/private/GigaLearnCPP/PPO/PPOLearner.cpp` — training loop + guiding-policy hook
- `GigaLearnCPP/src/public/GigaLearnCPP/PPO/PPOLearnerConfig.h` — PPO hyperparameters incl. `useGuidingPolicy`
- `GigaLearnCPP/src/public/GigaLearnCPP/PPO/TransferLearnConfig.h` — distillation config

Our stack's model config is in `ggl/train.cpp:142-150`: policy/critic `{512,512,512}`, ReLU, **`addLayerNorm = false`**, separate policy and critic (no shared head), Adam. Obs dim is **89** (verified by inspecting a live `POLICY.lt`, see Section 2).

## 1. BC training loop fundamentals

With `(obs, action_idx)` pairs from replay frames, BC is a plain supervised multi-class classification problem over the 90-action `DefaultAction` space.

**Canonical loop** (cross-entropy over logits):

```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader

model = make_policy()              # 89 -> 512 -> 512 -> 512 -> 90 (see §5)
opt   = torch.optim.Adam(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()    # expects raw logits, integer targets

for epoch in range(EPOCHS):
    for obs, act in DataLoader(ds, batch_size=300, shuffle=True):
        logits = model(obs.cuda())              # [B, 90]
        loss = loss_fn(logits, act.cuda().long())
        loss.backward(); opt.step(); opt.zero_grad()
```

### Hyperparameters — community reference

From [`Rolv-Arild/replay-pretraining`](https://github.com/Rolv-Arild/replay-pretraining) (`replay_pretraining/bcm/behavioral_cloning.py:40-58`):

| Knob        | Value |
|-------------|-------|
| Optimizer   | Adam (no weight decay) |
| LR          | `5e-5` |
| LR schedule | `LambdaLR(lambda e: 1 / (0.25 * e + 1))` — decays each epoch |
| Batch size  | 300 |
| Dropout     | 0.0 |
| Epochs      | 100 (but saves on best val loss, so ends earlier) |
| Loss        | `nn.CrossEntropyLoss()` on 90-way logits |
| Augmentation | None (replays are already mirrored by RLGym's `BLUE/ORANGE` symmetry if desired) |

Notes on this reference:

- Their actor is a **multi-branch architecture** (ball/boosts/agents branch + other-agents branch + `ControlsPredictorDot` head) — not a plain MLP. That design is tied to their obs builder. Do not copy the architecture; copy the training-loop hyperparameters, which are architecture-agnostic.
- LR `5e-5` with Adam is on the low side but is well-validated for RL-scale networks and 90-class classification. RL policies are easy to destabilize with high BC LRs because the logit scale matters for subsequent PPO clipping.
- No weight decay, no dropout. These are BC norms for RL because the dataset is large (tens of millions of frames) and you want the policy to memorize expert distributions, not generalize in the L2-regularized sense.

### Two critical dataset hygiene rules (learned from their code)

`replay_pretraining/bcm/bc_dataset.py:34-41`:

1. **Filter kickoff no-ops.** Kickoff frames frequently contain `action=no-op` while the player is waiting; if not filtered, the model learns "stand still at kickoff" and the warm-started bot sits there in-game. Filter with: `ball_pos == 0 & player_vel == 0 & action == no-op`.
2. **Filter stale physics ticks.** Replay files repeat physics values 2-3× per tick before fresh updates arrive. Only keep rows where the player's physics vector changed since the previous row: `(np.diff(x_data[:, PLAYER_PHYSICS_SLICE], axis=0, prepend=np.nan) != 0).any(axis=1)`.

Both filters come from `bc_dataset.py`. Apply them before training, not during.

### Action masking during BC

GGL's `DefaultAction` exposes `GetActionMask(player, gs)` (`RLGymCPP/ActionParsers/`). If a replay frame's target action is outside the current mask (e.g., dodge unavailable), options are:

- **Skip the frame.** Simplest. Upstream `replay-pretraining` does this implicitly — their label step only permits actions consistent with jump/dodge state (`README.md: "only permits actions matching available replay data"`).
- **Train with cross-entropy anyway.** Acceptable since at inference the PPO pipeline always applies the mask (`PPOLearner.cpp:89`, `ACTION_DISABLED_LOGIT = -1e10f`). But it's wasted gradient on logits the policy will never sample.

Recommendation: **skip frames where the target action is masked out**. Simpler and keeps the logit distribution clean on masked positions. [community claim, unverified — my read of common practice; worth a small ablation if you have capacity.]

### LR schedule, weight decay, augmentation — other approaches

No strong evidence of fancier schedules in Rocket-League-specific BC. Rolv-Arild's `1/(0.25*e+1)` is a reasonable default (roughly halves LR every 4 epochs). Cosine with warmup is the generic alternative. Weight decay is typically omitted in BC for RL warm-start because subsequent PPO will re-regularize via clip + entropy. Augmentation by BLUE/ORANGE mirroring doubles the effective dataset size — standard practice if teammate B does not already emit both perspectives.

## 2. GGL's checkpoint format

`Util/Models.cpp:89-99` — save:

```cpp
std::filesystem::path path = GetSavePath(folder);   // folder/POLICY.lt
auto streamOut = std::ofstream(path, std::ios::binary);
torch::save(seq, streamOut);                        // <-- libtorch C++ serialization
```

`Util/Models.cpp:101-128` — load:

```cpp
auto streamIn = std::ifstream(path, std::ios::binary);
torch::load(this->seq, streamIn, device);           // <-- matches torch::save
```

Where `seq` is `torch::nn::Sequential` built in `Models.cpp:16-29`:

```cpp
int lastSize = config.numInputs;                    // 89 for 1v1
for (int i = 0; i < config.layerSizes.size(); i++) {
    seq->push_back(torch::nn::Linear(lastSize, config.layerSizes[i]));
    if (config.addLayerNorm)
        seq->push_back(torch::nn::LayerNorm({(int64_t)config.layerSizes[i]}));
    lastSize = config.layerSizes[i];
    AddActivationFunc(seq, config.activationType);  // ReLU
}
if (config.addOutputLayer)
    seq->push_back(torch::nn::Linear(lastSize, config.numOutputs));   // 90
```

For our project (`ggl/train.cpp:142-150`, `addLayerNorm = false`) the concrete structure for `POLICY.lt` is exactly seven children:

```
[0] Linear(89, 512)
[1] ReLU
[2] Linear(512, 512)
[3] ReLU
[4] Linear(512, 512)
[5] ReLU
[6] Linear(512, 90)
```

`CRITIC.lt` is the same layout but with `Linear(512, 1)` final.

### What the file actually contains

Verified by opening a real `POLICY.lt` (`build-ggl/checkpoints_ggl/ggl_2/<ts>/POLICY.lt`) as a zip:

```
archive/data/0       182272   (512 * 89 * 4 bytes = Linear(89,512).weight fp32)
archive/data/1         2048   (Linear(89,512).bias)
archive/data/2      1048576   (Linear(512,512).weight)
archive/data/3         2048   (bias)
archive/data/4      1048576   (weight)
archive/data/5         2048   (bias)
archive/data/6       184320   (Linear(512,90).weight  -- 90*512*4)
archive/data/7          360   (90*4)
archive/data.pkl        873   (module tree + storage refs)
archive/code/__torch__.py + mangled submodules
archive/version = 3
archive/byteorder = little
```

Loaded via Python `torch.jit.load(...)`, the state dict is:

```
0.weight: torch.Size([512, 89])
0.bias:   torch.Size([512])
2.weight: torch.Size([512, 512])
2.bias:   torch.Size([512])
4.weight: torch.Size([512, 512])
4.bias:   torch.Size([512])
6.weight: torch.Size([90, 512])
6.bias:   torch.Size([90])
```

**So `torch::save(nn::Sequential, stream)` produces a TorchScript-style zip archive** — readable by `torch.jit.load` in Python. The key observation: `torch::load(seq, stream, device)` in libtorch (`Models.cpp:122`) reads parameters by the dotted-path keys `"0.weight"`, `"0.bias"`, `"2.weight"`, ... matching the positional indices of children in the `Sequential`. ReLU children (odd indices 1, 3, 5) have no parameters, so no keys for them.

### Safety check inside GGL's load path

`Models.cpp:119, 132-147` — GGL records parameter tensor sizes before load, reloads, then compares: if the saved model has different tensor shapes, it aborts with an explicit "Saved model has different size" error. This means:

- **If your Python-built module has shape mismatches (wrong obs dim, wrong layer sizes), GGL fails fast, won't silently succeed**. Good.
- If the parameter *names* (dotted-path keys) do not match, `torch::load` throws and is caught at `Models.cpp:123-128`, yielding a "checkpoint may be corrupt or of different model arch" error. Also good — fails cleanly.

## 3. Warm-start strategy

### Option A — build in Python, save TorchScript, drop into checkpoint dir (recommended)

Plan:

1. Build `nn.Sequential` in Python matching `ggl/train.cpp:142-150` exactly.
2. Train BC on `(obs, act)` pairs with cross-entropy.
3. Serialize via `torch.jit.script(net).save('POLICY.lt')`.
4. Create `checkpoints_ggl/<run-name>/0/` folder containing `POLICY.lt` plus a placeholder `CRITIC.lt` (fresh-random Sequential with same shape, `Linear(512, 1)` output) plus a `RUNNING_STATS.json` (see §5).
5. Launch GGL with `GGL_RUN_NAME=<run-name>`; `Learner::Load()` picks the highest-numbered timestep folder (`Learner.cpp:255-268`) and calls `ppo->LoadFrom(...)`, which loads each `.lt` via `Models::Load()`.

**Known compatibility risk.** Internal inspection shows two differences between `torch::save(seq)` archives and `torch.jit.script(seq).save()` archives:

| | GGL archive | Python-scripted archive |
|--|--|--|
| top-level module class | `__torch__.Module` (anonymous) | `__torch__.torch.nn.modules.container.Sequential` |
| zip prefix | `archive/` | `MODEL_NAME/` (stem of save path) |
| param keys | `0.weight`, `0.bias`, ... | `0.weight`, `0.bias`, ... (same) |
| pickle `version` | 3 | 3 (same) |

`torch::load(nn::Sequential&, stream, device)` in libtorch calls `serialize::InputArchive::load_from` then `Module::load()` which walks `named_parameters()` by dotted key path (`0.weight`, `2.bias`, etc.). The parameter key paths are identical between the two formats. The top-level module class name is stored in `data.pkl` but ignored when loading into a pre-constructed `Sequential` — libtorch does not re-instantiate the module; it fills the one you already built. On those grounds the load **should** succeed.

I was unable to directly exercise `torch::load(seq, streamIn)` from Python — this is the one thing that cannot be cheaply verified without writing a small C++ test harness. The risk is low but non-zero. Mitigation in §5.

### Option B — only train the final logits head

Only train `seq[6]` (`Linear(512, 90)`) on a frozen randomly-initialized backbone. Fast, simple — but you're cloning a random feature extractor's outputs, so you only learn action frequencies, not good representations. Not worth it unless dataset is tiny. Most Rocket League BC work trains the full network.

### Option C — PPO with a KL / L1 regularizer to a frozen BC policy

GGL already ships this as `useGuidingPolicy` (`PPOLearnerConfig.h:55-57` + `PPOLearner.cpp:224-235`):

```cpp
// PPOLearner.cpp:224-235
if (config.useGuidingPolicy) {
    torch::Tensor guidingProbs;
    { RG_NO_GRAD; guidingProbs = InferPolicyProbsFromModels(guidingPolicyModels, obs, actionMasks, ...); }
    auto guidingLoss = (guidingProbs - probs).abs().mean();   // L1, not KL
    avgGuidingLoss.Add(guidingLoss.detach().cpu().item<float>());
    guidingLoss = guidingLoss * config.guidingStrength;       // default 0.03
    ppoLoss = ppoLoss + guidingLoss;
}
```

This is **L1 on action-probability differences**, not true KL. It is a regularizer against catastrophic forgetting of BC priors during PPO, applied every gradient step. Config surface:

- `cfg.ppo.useGuidingPolicy = true;`
- `cfg.ppo.guidingPolicyPath = "<path to dir with frozen BC POLICY.lt>";`
- `cfg.ppo.guidingStrength = 0.03f;`  // tune down over time; good early, costly late

Separately, `Learner::StartTransferLearn()` (`Learner.cpp:289-460`) does a purer distillation run with `TransferLearnConfig::useKLDiv = true` (true KL), but that runs as a distinct training mode (`StartTransferLearn()` replaces `Start()`), not during normal PPO.

### What the community actually uses

Rolv-Arild's Necto lineage: "Inspired by Video PreTraining, we can now use replay files to learn from human gameplay and see years of gameplay before setting a wheel on the field" — their pipeline is BC pretrain → PPO fine-tune. No explicit KL-to-BC regularizer documented in their public repos. [community claim, unverified.] The Nexto release notes mention it was "further trained extensively with reinforcement learning" after the BC phase.

**Recommended stack for this project**: Option A + Option C. BC pretrain, drop `POLICY.lt` into a fresh run folder, then run GGL with `useGuidingPolicy=true` pointing at a frozen copy of the same `POLICY.lt` to prevent early PPO drift.

## 4. Post-warm-start training

### The critic is random at load time

GGL's `Model::Load` accepts `allowNotExist=true` (`Models.h:188-191`, `Models.cpp:104-110`), so if you omit `CRITIC.lt` you get a warning but the critic stays at its random init. Either way, the critic is essentially random relative to the BC policy. Consequences:

- First few PPO iterations will produce nonsense advantages (critic predictions are random).
- If the actor steps on those advantages with full LR, it can unlearn BC immediately.

Mitigations:

- **Critic warm-up phase.** Freeze the actor (`cfg.ppo.policyLR = 0`) for N iterations so the critic catches up with actor-induced state distribution. GGL handles `policyLR == 0` cleanly (`PPOLearner.cpp:161-164`: `bool trainPolicy = config.policyLR != 0`). After ~1M timesteps of critic-only training, re-enable actor.
- **Critic BC target.** Independently harder; requires computing discounted Monte-Carlo returns from replay rewards. Not recommended for a first pass — the in-sim critic warm-up achieves the same effect in ~1M steps.

### Entropy coefficient during warm-start

`cfg.ppo.entropyScale` defaults to `0.018f` (`PPOLearnerConfig.h:39`); our `train.cpp` sets `0.01f` (line 138). With a BC-trained actor, the entropy is already low (BC collapses exploration), so the normalized entropy term pushes the actor toward uniform. Two recommendations:

- **Keep** `entropyScale` low early (matches our 0.01) so you don't forcibly raise entropy and unlearn BC. Raise it later if exploration stalls.
- Or briefly set it lower (~0.001) during the first 1-2M timesteps, then restore.

No `Learner::SetEntropyScale` exposed — you'd need to edit the config between runs. Accept the default for first pass.

### Preventing catastrophic forgetting

Use `useGuidingPolicy = true` pointing at the frozen BC `POLICY.lt`, with `guidingStrength = 0.03` (GGL default). This adds an L1-prob-diff regularizer (`PPOLearner.cpp:231-234`). The strength is a hyperparameter; start at 0.03 and decay to 0 over ~10-50M timesteps. GGL doesn't expose decay automatically — anneal by editing config and resuming, or leave flat.

## 5. Concrete recipe

Goal: given ~1M `(obs, action_idx)` pairs from teammate B, produce `checkpoints_ggl/bc-warm/0/POLICY.lt` that `ggl/train.cpp`'s `Learner` loads on startup.

### Python code

```python
# ggl/python_scripts/bc_train.py
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

OBS_DIM    = 89
N_ACTIONS  = 90
LAYERS     = [512, 512, 512]

def make_policy():
    """Match ggl/train.cpp:142-150 exactly: ReLU, no LayerNorm, separate actor."""
    layers = []
    last = OBS_DIM
    for h in LAYERS:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers += [nn.Linear(last, N_ACTIONS)]
    return nn.Sequential(*layers)

def train(obs, act, epochs=30, batch=300, lr=5e-5, device="cuda"):
    model = make_policy().to(device)
    ds = TensorDataset(torch.from_numpy(obs).float(), torch.from_numpy(act).long())
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1/(0.25*e + 1))
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        running = 0.0; n = 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item(); n += 1
        sched.step()
        print(f"epoch {epoch}: train_loss={running/n:.4f}  lr={sched.get_last_lr()[0]:.2e}")
    return model

def save_ggl_checkpoint(model, critic_template, checkpoint_dir):
    """checkpoint_dir = build-ggl/checkpoints_ggl/bc-warm/0/"""
    import os, json, pathlib
    pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    torch.jit.script(model.cpu()).save(os.path.join(checkpoint_dir, "POLICY.lt"))
    torch.jit.script(critic_template.cpu()).save(os.path.join(checkpoint_dir, "CRITIC.lt"))
    # RUNNING_STATS.json with the keys Learner::LoadStats reads (Learner.cpp:185-209)
    stats = {"total_timesteps": 0, "total_iterations": 0}
    # obs_stat / return_stat only required if cfg.standardizeObs / cfg.standardizeReturns are set;
    # our train.cpp leaves them off by default, so we can omit.
    with open(os.path.join(checkpoint_dir, "RUNNING_STATS.json"), "w") as f:
        json.dump(stats, f, indent=4)

def make_critic_template():
    layers = []
    last = OBS_DIM
    for h in LAYERS:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers += [nn.Linear(last, 1)]           # Linear(512, 1) final
    return nn.Sequential(*layers)

if __name__ == "__main__":
    obs = np.load("bc_obs.npy")    # shape (N, 89), float32
    act = np.load("bc_act.npy")    # shape (N,), int64, in [0, 90)
    assert obs.shape[1] == OBS_DIM and act.max() < N_ACTIONS
    model  = train(obs, act)
    critic = make_critic_template()
    save_ggl_checkpoint(model, critic, "build-ggl/checkpoints_ggl/bc-warm/0/")
```

### Launch GGL with the warm-started actor

```sh
export GGL_RUN_NAME=bc-warm
cd build-ggl
./rl_rl_ggl
```

On startup, `Learner::Load()` (`Learner.cpp:249-269`) picks `checkpoints_ggl/bc-warm/0/` and calls `ppo->LoadFrom(...)` → `models.Load(folder, allowNotExist=true, loadOptim=true)`. Missing optimizer files trigger a warning and reset — fine, first run.

If `torch::load` rejects the Python-scripted archive (see compatibility risk in §3 Option A), you will see one of:

- `"Failed to load model "policy", checkpoint may be corrupt or of different model arch."` — means libtorch's deserializer rejected the archive layout.
- `"Saved model has different size..."` — means shapes mismatch; re-check that obs dim in runtime matches `OBS_DIM=89` (it's `envSet->state.obs.size[1]` at `Learner.cpp:91`).

Fallback if archive layout is rejected: save a **state dict** via `torch.save(model.state_dict(), ...)` and write a tiny C++ helper that instantiates a `GGL::Model` with matching `ModelConfig`, copies the params in manually via `model->parameters()`, and calls `model->Save(folder)`. That produces a canonical libtorch archive. Roughly:

```cpp
// ggl/python_scripts/.. actually a one-shot C++ binary linked against GGL
auto m = new GGL::Model("policy", cfgPolicy, torch::kCPU);
// read state_dict from disk (torch.load pickled dict), copy in name-matched order
for (auto& p : m->parameters()) { p.data().copy_(loaded_tensor); }
m->Save(save_dir, /*saveOptim=*/false);
```

This round-trip is guaranteed-compatible.

### Enable guiding-policy regularization for PPO

After the first warm-started run, copy the same `POLICY.lt` to `build-ggl/guiding_policy/POLICY.lt` and set in `train.cpp`:

```cpp
cfg.ppo.useGuidingPolicy  = true;
cfg.ppo.guidingPolicyPath = "guiding_policy";
cfg.ppo.guidingStrength   = 0.03f;
```

Rebuild and relaunch. `PPOLearner.cpp:32-36` loads it at learner construction; the regularizer is applied every minibatch.

### Validation

1. **Smoke-test the BC model in isolation**, *before* hooking into GGL:

   ```python
   m = torch.jit.load("bc-warm/0/POLICY.lt")
   x = torch.randn(4, 89)
   # Invoke children manually; the saved Sequential may lack a top-level forward
   for child in m.children(): x = child(x)
   print(x.shape)   # [4, 90]
   ```

   If this works, the policy is structurally valid regardless of whether GGL loads it.

2. **Load validation via GGL**. Easiest: set `cfg.renderMode = true;` in `train.cpp` (actually, set via env var `GGL_RENDER=1`), launch GGL, watch the car in rlviser. A BC-warmed policy should move purposefully toward the ball and attempt touches even before PPO starts — unlike a fresh model which will drive randomly.

3. **Action-distribution sanity**. Compare the BC model's action-index histogram on a validation slice to the replay action histogram. They should match closely; if the BC model collapses to one or two actions, something is wrong (usually: no-ops at kickoff not filtered, or the LR is too high).

4. **Per-action accuracy**. On val set, top-1 accuracy of ~20-30% is normal for 90-way Rocket League action classification — human actions are multimodal and small variations map to different indices. Top-5 accuracy is a better metric; expect 50-60%. [community claim, unverified, based on Rolv-Arild's reported training curves.]

## References

- GGL source (this repo copy): `/home/seis/code/GigaLearnCPP-Leak/GigaLearnCPP/src/`
- Our project's training entrypoint: `ggl/train.cpp:142-150` (network config) and `ggl/train.cpp:155-163` (checkpoint path).
- Live checkpoints for format inspection: `/home/seis/code/rl_rl/build-ggl/checkpoints_ggl/ggl_2/<timestep>/POLICY.lt`
- [`Rolv-Arild/replay-pretraining`](https://github.com/Rolv-Arild/replay-pretraining) — reference BC pipeline for Rocket League (`replay_pretraining/bcm/behavioral_cloning.py` + `bc_dataset.py`).
- [`Rolv-Arild/Necto`](https://github.com/Rolv-Arild/Necto) — umbrella repo describing the BC→PPO training philosophy.
- [`AechPro/rlgym-ppo`](https://github.com/AechPro/rlgym-ppo) — Python PPO reference that GGL mirrors (noted in `PPOLearnerConfig.h:7`).
