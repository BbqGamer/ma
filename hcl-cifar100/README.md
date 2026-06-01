# Reproduction: Hierarchical Class-Based Curriculum Loss
Authors: Palash Goyal, Shalini Ghosh

Reproduction is done on the CIFAR-100 dataset, which is different than the ones that
were used in the original paper: Diatoms, IMCLEF

The goal is to check whether by changing the loss from CrossEntropy to HCL we can
observe better results in Hit@1, MRR and most notably HierDist.

The training is supposed to be containerized so that we can easily run it on runpod.

