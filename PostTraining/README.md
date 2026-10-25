# Post-training examples

The repository README describes the complete workflow and setup. This directory contains the NumPy reference contracts, explicit PyTorch loss implementation, synthetic arithmetic inputs, optional CUDA SFT/DFT smoke entry, recovery helpers and their offline tests.

SFT uses target-token NLL. DFT applies `exp(-nll.detach()) * nll`, normalized by the same target-token count. A lower DFT optimization loss is not evidence of better task performance. The two branches share seed, data order, resolved model revision and adapter initialization checks.

Notebook installation/training switches are off. The CPU examples and failure/timeout tests do not download models. `experiments/requirements-colab.txt` is an unvalidated candidate training environment; real CUDA training requires separate setup and authorization. Synthetic samples test plumbing rather than generalization.
