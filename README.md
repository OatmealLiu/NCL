# ResTune Benchmark
Supervise-pretrained model w/o HNG

## CIFAR-10
- [x] Warmup
- [x] INCD
```shell
sbatch -p gpupart -c 4  --gres gpu:1 ./scripts/ncl_sl_cifar10.sh
```

## CIFAR-100
- [x] Warmup
- [x] INCD
```shell
sbatch -p gpupart -c 4  --gres gpu:1 ./scripts/ncl_sl_cifar100.sh
```

## Tiny-ImageNet
- [x] Warmup
- [x] INCD
```shell
sbatch -p gpupart -c 4  --gres gpu:1 ./scripts/ncl_sl_tinyimagenet.sh
```

## 2-step INCD auto_run
- [x] CIFAR-100
- [x] TinyImageNet

```shell
sbatch -p gpupart -c 4  --gres gpu:1 ./ablation_two_scripts/auto_cifar100_two_step.sh
sbatch -p gpupart -c 4  --gres gpu:1 ./ablation_two_scripts/auto_tiny_two_step.sh
```
scp -r ablation_two_scripts zhun@todi.disi.unitn.it:/data/zhun/NCL
scp two_NCL_incd_train_cifar100.py zhun@todi.disi.unitn.it:/data/zhun/NCL
scp two_NCL_incd_train_tinyimagenet.py zhun@todi.disi.unitn.it:/data/zhun/NCL

scp resnet.py zhun@todi.disi.unitn.it:/data/zhun/NCL/models/
scp fair_evals.py zhun@todi.disi.unitn.it:/data/zhun/NCL/utils/
