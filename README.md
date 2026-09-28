# TDV-LeWM: Temporal-Difference Supervision for Latent World-Model Planning

Applying TDV-style motion learning to LeWM on OGBench-Cube.

[Project page](https://flamma7.github.io/tdv-lewm/) · [Model weights](https://huggingface.co/flamma77/lewm-base)

<video src="./docs/assets/combined_9.mp4" width="100%" autoplay muted loop playsinline controls></video>

## Approach

TDV-LeWM: I augmented LeWorldModel with a TDV-style motion encoder to bias the latent space away toward motion and task-relevant geometry. The motion encoder maps \(\Delta x_t\) with cross-attention to \(H_t\), and the residual \(\Delta z_t\) is trained so that \(z_t + \Delta z_t \approx z_{t+1}\). I also analyzed the OGBench-Cube dataset and discovered ~38% of this popular benchmark is trivially successful, inflating MPC performance metrics.

$$(z_t, H_t) = \mathrm{Enc}(x_t), \qquad \Delta z_t = m_\phi(\Delta x_t, H_t), \qquad \mathcal{L}_{\mathrm{TDV}} = \lVert z_t + \Delta z_t - z_{t+1} \rVert^2$$

$$\mathcal{L} = \mathcal{L}_{\mathrm{pred}} + \lambda\,\mathcal{L}_{\mathrm{SIGReg}} + \alpha\,\mathcal{L}_{\mathrm{TDV}}$$

- **Model.** ViT-T/14 motion encoder on \(\Delta x_t\), cross-attending to \(H_t\). Predictor unchanged from LeWM. Motion encoder discarded at planning time. SIGReg instead of TDV’s DINO teacher.
- **Data.** OGBench-Cube (single cube) through [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel). Frameskip 5, horizon 25.
- **Training.** 10 epochs, batch size 128, learning rate \(5\times 10^{-5}\), on RTX 5090s via RunPod. Best run: \(\alpha=1.0\), \(\lambda=0.25\).
- **Evaluation.** CEM, iCEM, and Adam anytime success on the cube, the gripper, and both (4 cm).

## Repo layout

```
scripts/train/lewm_tdv.py         # TDV-LeWM training (source of reported results)
scripts/train/lewm.py             # LeWM reproduction (LeWM_REP)
scripts/train/lewm_visreg.py      # VISReg in place of SIGReg
scripts/plan/eval_wm_cube_mpc.py  # CEM / iCEM / Adam anytime success
scripts/plan/eval_wm_cube_plan.py # latent cost vs cube and gripper outcomes
scripts/plan/eval_straightness.py # consecutive latent-velocity cosine
controller.py                     # dispatch train / mpc / plan jobs
deploy.py                         # RunPod pods; recycle on bad NVIDIA drivers
job_configs/                      # sweep specs (tdv, lewm, visreg)
analysis/                         # no-op counts, MPC tables, ranking, straightness
docs/                             # project page
```

Reported TDV-LeWM numbers are from `scripts/train/lewm_tdv.py` (\(\alpha=1.0\), \(\lambda=0.25\)). `lewm.py` is \(\mathrm{LeWM_{REP}}\). `controller.py` reads a `job_configs/*.yaml` and launches training or eval on RunPod; checkpoints go to [Hugging Face](https://huggingface.co/flamma77/lewm-base).

## Learnings

- Cube no-ops are ~38% of the eval set and account for most of the inflation in cube and both-object success. Once they are removed, success rate drops considerably for LeWM, and the performance improvement of TDV-LeWM in comparison widens.
- Every model scores the gripper much higher than the cube. The gripper occupies more of the image, so it takes a larger share of the latent that MPC optimizes. Most cube no-ops still require the gripper to move (the joint both-no-op rate is only ~4%).
- TDV-LeWM needed a larger SIGReg weight (\(\lambda=0.25\) vs \(0.1\)) to offset the extra predictive bias from \(\mathcal{L}_{\mathrm{TDV}}\). The published checkpoint also keeps a slightly higher Roy–Vetterli effective rank than either model I trained.
- TDV ranks cube outcomes and the combined cube–gripper objective best (pooled Spearman \(\rho_{\mathrm{cube}}=0.608\), \(\rho_{\mathrm{cg}}=0.657\), positive \(\rho_{\mathrm{cg}}\) on 94% of scenarios). Gripper outcomes are ranked more faithfully than cube outcomes for all three models.
- Learning the one-step displacement \(z_{t+1} \approx z_t + \Delta z_t\) does not straighten latent trajectories. Mean consecutive-velocity cosine is 0.661 for TDV-LeWM, against 0.674 and 0.669 for \(\mathrm{LeWM_{REP}}\) and \(\mathrm{LeWM_{PUB}}\).

## References

1. Ninad Daithankar, Alexi Gladstone, Yann LeCun, and Heng Ji. *You Don’t Need Strong Assumptions: Visual Representation Learning via Temporal Differences*. arXiv, 2026. [arXiv:2606.15956](https://arxiv.org/abs/2606.15956)
2. Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*. arXiv, 2026. [arXiv:2603.19312](https://arxiv.org/abs/2603.19312)
3. Seohong Park, Kevin Frans, Benjamin Eysenbach, and Sergey Levine. *OGBench: Benchmarking Offline Goal-Conditioned RL*. ICLR, 2025. [arXiv:2410.20092](https://arxiv.org/abs/2410.20092)
4. Ying Wang, Oumayma Bounou, Gaoyue Zhou, Randall Balestriero, Tim G. J. Rudner, Yann LeCun, and Mengye Ren. *Temporal Straightening for Latent Planning*. arXiv, 2026. [arXiv:2603.12231](https://arxiv.org/abs/2603.12231)
5. Randall Balestriero and Yann LeCun. *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*. arXiv, 2025. [arXiv:2511.08544](https://arxiv.org/abs/2511.08544)
