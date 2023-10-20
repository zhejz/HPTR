# Configurations of Ablation Models

Use the following configurations and adapt [bash/train.sh](../bash/train.sh) to train the ablation models.

## Input Representation
- Our `HPTR` for Waymo dataset. The model has 15.2M parameters.
  ```
  model=scr_womd \
  ```
- Our `HPTR` for AV2 dataset.
  ```
  trainer=av2 \
  model=scr_av2 \
  datamodule=h5_av2 \
  ```
- Agent-centric baseline [Wayformer](https://arxiv.org/abs/2207.05844), i.e. `WF baseline`.
  ```
  model=acg_womd \
  ```
- Scene-centric baseline [SceneTransformer](https://arxiv.org/abs/2106.08417), i.e. `HPTR SC`.
  ```
  model=scg_womd \
  ```

## Hierarchical Architecture

- `HPTR diag+full` with 15.4M parameters. It needs RTX 3090 for training.
  ```
  model.model.intra_class_encoder.n_layer_tf_map=6 \
  model.model.intra_class_encoder.n_layer_tf_tl=2 \
  model.model.intra_class_encoder.n_layer_tf_agent=2 \
  model.model.decoder.tf_n_layer=2 \
  model.model.decoder.k_reinforce_tl=-1 \
  model.model.decoder.k_reinforce_agent=-1 \
  model.model.decoder.k_reinforce_all=1 \
  ```
- `HPTR diag` with 15.4M parameters.
  ```
  model.model.intra_class_encoder.n_layer_tf_map=6 \
  model.model.intra_class_encoder.n_layer_tf_tl=3 \
  model.model.intra_class_encoder.n_layer_tf_agent=3 \
  model.model.decoder.tf_n_layer=2 \
  model.model.decoder.k_reinforce_tl=-1 \
  model.model.decoder.k_reinforce_agent=-1 \
  ```
- `HPTR full` with 15.2M parameters. It needs RTX 3090 for training.
  ```
  model.model.intra_class_encoder.n_layer_tf_map=-1 \
  model.model.decoder.tf_n_layer=6 \
  model.model.decoder.k_reinforce_tl=-1 \
  model.model.decoder.k_reinforce_agent=-1 \
  model.model.decoder.k_reinforce_all=1 \
  ```

## Others
- Different polyline embedding.
  ```
  model.pre_processing.relative.pose_pe.agent=xy_dir \
  model.pre_processing.relative.pose_pe.map=xy_dir \
  ```
- Attention without bias.
  ```
  model.model.tf_cfg.bias=False \
  ```
- Different RPE mode.
  ```
  model.model.rpe_mode=xy_dir \
  model.model.rpe_mode=pe_xy_dir \
  ```
- Apply RPE to query. It needs RTX 3090 for training.
  ```
  model.model.tf_cfg.apply_q_rpe=True \
  ```
- Without anchor reinforce (17.5M parameters).
  ```
  model.model.decoder.tf_n_layer=3 \
  model.model.decoder.k_reinforce_agent=8 \
  model.model.decoder.k_reinforce_anchor=-1 \
  ```
- Without anchor reinforce, larger model (23.3 parameters).
  ```
  model.model.n_tgt_knn=50 \
  model.model.intra_class_encoder.n_layer_tf_map=6 \
  model.model.decoder.tf_n_layer=4 \
  model.model.decoder.k_reinforce_agent=8 \
  model.model.decoder.k_reinforce_anchor=-1 \
  ```