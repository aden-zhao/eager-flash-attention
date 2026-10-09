# Source on Perlmutter before building or running:  source perlmutter/env.sh
module load cudatoolkit/13.2 nccl/2.29.2-cu13 cmake >/dev/null 2>&1
export MPICH_GPU_SUPPORT_ENABLED=1
export SLURM_ACCOUNT=${SLURM_ACCOUNT:-m4999_g}
# NCCL 2.29.2 + the libfabric plugin hang on inter-node AllReduces of <= 512 KB with the
# LL/LL128 protocols (observed 2026-09-14; larger messages and single-node runs are unaffected).
# Forcing the Simple protocol avoids it. The multi-node benchmark tables were measured with the
# default protocol selection, which never picked LL for those message sizes.
export NCCL_PROTO=Simple
