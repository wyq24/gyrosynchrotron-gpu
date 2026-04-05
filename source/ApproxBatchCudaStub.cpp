#include "ApproxBatchInternal.h"

int MWApproxBatchRunCuda(const MWApproxBatchConfig *, const MWApproxBatchInputs *, const MWApproxBatchOutputs *)
{
 return MW_BATCH_ERR_CUDA_UNAVAILABLE;
}

int MWApproxBatchCudaAvailableImpl(void)
{
 return 0;
}
