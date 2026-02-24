### Run training

```bash
python train.py --data_path ./data_scene_flow/training --epochs 100 --batch_size 8 --lr 1e-4
```

On the first run you will see:
```
Starting from scratch
Epoch 1/100: 100%|████████| 97/97 [05:23<00:00]
Epoch 1/100 — Total: 0.5028 | Scale 1: 0.2452 | Scale 2: 0.1284 | Scale 3: 0.0773 | Scale 4: 0.0519
```

On subsequent runs, training resumes automatically from the last checkpoint:
```
Restored from ./checkpoints/ckpt-1
Epoch 6/100: ...
```

| Scale | Weight |
|---|---|
| Scale 1 (finest) | 1.0 |
| Scale 2 | 0.5 |
| Scale 3 | 0.25 |
| Scale 4 (coarsest) | 0.125 |

The smoothness loss is weighted by `smooth_weight=0.1` relative to the photometric loss.

---