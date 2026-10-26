# Pinned upstream sources

Repository: https://github.com/sstary/SSRS

Commit: `b89998178ab07679c179fbda3b28eefcd0e361eb`

The source files are distributed under the upstream Apache-2.0 license, preserved
in `LICENSE`. Original project citations and usage are preserved in
`UPSTREAM_README.md`. No model parameters or datasets are included.

| Original path | Original SHA-256 |
| --- | --- |
| `RS3Mamba/model/RS3Mamba.py` | `c18bc6f7e41a84d479e52275a090e80debca0335468cc75ee9542593d6c06862` |
| `RS3Mamba/model/SwinUMamba.py` | `78a9c7370167b43a8a7bc41a73e50981bd52903c264b072d8153826887a40157` |

Source modifications are marked at the top of each file:

- `RS3Mamba.py`: changed `model.SwinUMamba` to `.SwinUMamba` for package imports.
- `SwinUMamba.py`: removed two unused MONAI imports (`UnetOutBlock`,
  `UnetrBasicBlock`, `UnetrUpBlock`). None of these symbols is used in this file.
- Trailing whitespace was removed from the two Python files and the copied
  upstream README; line endings are normalized to LF in Git.

The original forward computation, stage depths, selective scan, CCM fusion and
decoder are preserved. The project wrapper handles explicit local checkpoint
loading and GF-1 input-band adaptation. It does not call the upstream permissive
checkpoint helper. Its strict loader reproduces the upstream skip list, including
loading `patch_embed.norm.bias` while leaving `patch_embed.norm.weight` at its
default initialization (the upstream skip list repeats the latter).

The CNN SWSL checkpoint has its own CC-BY-NC 4.0 terms; it is downloaded separately
from Meta's public model host and is not redistributed in this Git repository:
https://github.com/facebookresearch/semi-supervised-ImageNet1K-models
