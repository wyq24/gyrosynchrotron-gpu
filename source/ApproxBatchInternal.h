#pragma once

#include "ApproxBatch.h"

int MWApproxBatchValidate(const MWApproxBatchConfig *config, const MWApproxBatchInputs *inputs, const MWApproxBatchOutputs *outputs);
void MWApproxBatchFillFrequencyGrid(const MWApproxBatchConfig *config, double *freq_hz);
int MWApproxBatchRunCpu(const MWApproxBatchConfig *config, const MWApproxBatchInputs *inputs, const MWApproxBatchOutputs *outputs);
int MWApproxBatchRunCuda(const MWApproxBatchConfig *config, const MWApproxBatchInputs *inputs, const MWApproxBatchOutputs *outputs);
int MWApproxBatchCudaAvailableImpl(void);
void MWApproxBatchTimingResetImpl(void);
void MWApproxBatchTimingStore(const MWApproxBatchTiming *timing);
