#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
cat <<'EOF'
Audited BAGEL files:
- train/pretrain_unified_navit.py
- data/dataset_base.py
- data/t2i_dataset.py
- data/interleave_datasets/edit_dataset.py
- data/interleave_datasets/interleave_t2i_dataset.py
- data/optional_plan_dataset.py
- data/dataset_info.py
- inferencer.py
- image_edit.py
- eval/gen/gen_images_mp_imgedit.py
EOF
