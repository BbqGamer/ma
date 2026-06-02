# Reproduction: Hierarchical Class-Based Curriculum Loss
Authors: Palash Goyal, Shalini Ghosh

Reproduction is done on the CIFAR-100 dataset, which is different than the ones that
were used in the original paper: Diatoms, IMCLEF

The goal is to check whether by changing the loss from CrossEntropy to HCL we can
observe better results in Hit@1, MRR and most notably HierDist.

The training is supposed to be containerized so that we can easily run it on runpod.

## Docker image workflow

Versioned image builds are managed through `VERSION` and a single release command:

```bash
make release
```

By default this pushes to:

```bash
bbqdocker/hcl-cifar100
```

with tags:

```bash
bbqdocker/hcl-cifar100:v$(cat VERSION)-$(git rev-parse --short HEAD)
bbqdocker/hcl-cifar100:latest
```

Optional overrides:

```bash
IMAGE=yourdockerhubuser/hcl-cifar100 make release
PLATFORM=linux/amd64 make release
```

