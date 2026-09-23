# Third-Party Notices

The root `LICENSE` applies to original material in this repository. It does not
replace the following upstream notices.

## SirenFNO

- Project: SirenFNO
- Public upstream: <https://github.com/pengqingshi/SirenFNO.git>
- Pinned commit: `81918ecce323a2fd5c5a54db917598bda088574b`
- Included as: unmodified Git submodule at `third_party/SirenFNO`
- Used directories/files: `neuralop/`, `baseline/AMFNO.py`,
  `baseline/UFNO.py`, and `SirenFNO2D.py`
- Local modifications inside the submodule: none
- License: MIT
- Copyright: Copyright (c) 2026 Pengqing Shi

The upstream `LICENSE` is preserved at `third_party/SirenFNO/LICENSE`. The
source is identified by the pinned Git commit; selected canonical Git blob IDs
are also recorded in `third_party/UPSTREAM_VERSIONS.json` so verification is
independent of checkout line endings.

### SirenFNO MIT notice

MIT License

Copyright (c) 2026 Pengqing Shi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Apache-2.0 components included through SirenFNO

The following files carry Apache License, Version 2.0 notices:

- `third_party/SirenFNO/neuralop/mpu/comm.py`
- `third_party/SirenFNO/neuralop/mpu/helpers.py`
- `third_party/SirenFNO/neuralop/mpu/mappings.py`

Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.

These files are included through the pinned SirenFNO snapshot at commit
`81918ecce323a2fd5c5a54db917598bda088574b`. Their original copyright and
license headers are retained. The complete license text is provided in
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

This notice identifies the license applicable to the listed files. It does not
change the license of CAFE+FNO as a whole or SirenFNO as a whole, and it does
not replace the licenses or notices applicable to other components of this
distribution.

## NeuralOperator bundled with SirenFNO

- Import package: `neuralop`
- Declared version: `1.0.2`
- Distribution: complete `neuralop/` tree in the pinned SirenFNO submodule
- Git tree: `fbc6aa738d6ffc2e2ca6808e724b0083c3a688eb`
- Included path: `third_party/SirenFNO/neuralop`
- Local modifications: none
- License: MIT
- Copyright: Copyright (c) 2023 NeuralOperator developers

The bundled license is preserved at
`third_party/SirenFNO/neuralop/LICENSE`. Paper experiments import FNO/TFNO,
Trainer, losses, loaders, AdamW, and utilities directly from this tree and
reject other installed `neuralop` distributions.

### NeuralOperator MIT notice

MIT License

Copyright (c) 2023 NeuralOperator developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
