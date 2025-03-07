# FLA-TILELANGs
We aim at writing all [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) triton kernels in [TileLang](https://github.com/tile-ai/tilelang) for better performance.

# Install
```
pip install tilelang=0.1.2
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```
> **Note for H100 users**: Triton nightly version is required to avoid errors. See [issue #196](https://github.com/fla-org/flash-linear-attention/issues/196) for details.
