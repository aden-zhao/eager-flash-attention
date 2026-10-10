#pragma once

#include "attn_defs.cuh"
#include "attn_main.cuh"

enum RunMode {
    MODE_ATTN = 0,        // attention only, one launch over all S (kernel cost floor)
    MODE_ATTN_TILED = 1,  // attention only, one launch per tile (tiling/launch overhead)
    MODE_MEGATRON = 2,    // attention -> projection -> one AllReduce, serialized
    MODE_SYNC = 3,        // per tile: attention, projection, AllReduce with host syncs (no overlap)
    MODE_OVERLAP = 4,     // per tile, event-chained across three streams, nbuf tiles in flight
    MODE_STREAM = 5,      // one attention launch; projection + AllReduce stream behind it per tile,
                          // gated by device-side tile counters (cuStreamWaitValue32)
    MODE_COMM = 6,        // AllReduce only: num_tiles serialized AllReduces of tile x d floats
                          // (for fitting a latency + bytes/bandwidth cost model)
    MODE_COUNT = 7
};

const char* mode_name(int mode);
int parse_mode(const char* s);   // -1 if unknown

// All drivers write the full [S][d] reduced output to `output` (except the
// attention-only modes, which leave it untouched) and return the wall time in
// ms, maxed across ranks.
double run_mode(int mode, const Weights& w, float* output,
                   const AttnParams& p, const RunCtx& c);
