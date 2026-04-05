#include <math.h>
#include <string.h>
#include "ApproxBatchInternal.h"

namespace
{

MWApproxBatchTiming g_last_batch_timing{};

}

void MWApproxBatchTimingResetImpl(void)
{
 memset(&g_last_batch_timing, 0, sizeof(g_last_batch_timing));
}

void MWApproxBatchTimingStore(const MWApproxBatchTiming *timing)
{
 if (!timing)
 {
  MWApproxBatchTimingResetImpl();
  return;
 }
 g_last_batch_timing=*timing;
}

void MWApproxBatchTimingReset(void)
{
 MWApproxBatchTimingResetImpl();
}

void MWApproxBatchTimingGet(MWApproxBatchTiming *timing)
{
 if (!timing) return;
 *timing=g_last_batch_timing;
}

int MWApproxBatchValidate(const MWApproxBatchConfig *config, const MWApproxBatchInputs *inputs, const MWApproxBatchOutputs *outputs)
{
 if (!config || !inputs || !outputs) return MW_BATCH_ERR_INVALID_ARGUMENT;
 if (!outputs->status || !outputs->freq_hz || !outputs->jx || !outputs->kx || !outputs->jo || !outputs->ko)
  return MW_BATCH_ERR_INVALID_ARGUMENT;
 if (!inputs->area_cm2 || !inputs->depth_cm || !inputs->bmag_g || !inputs->temperature_k ||
     !inputs->thermal_density_cm3 || !inputs->nonthermal_density_cm3 || !inputs->delta ||
     !inputs->theta_deg || !inputs->emin_mev || !inputs->emax_mev)
  return MW_BATCH_ERR_INVALID_ARGUMENT;
 if (config->backend != MW_BATCH_BACKEND_CPU && config->backend != MW_BATCH_BACKEND_CUDA)
  return MW_BATCH_ERR_UNSUPPORTED_BACKEND;
 if (config->precision != MW_BATCH_PRECISION_FP64 && config->precision != MW_BATCH_PRECISION_FP32)
  return MW_BATCH_ERR_INVALID_ARGUMENT;
 if (config->batch_size <= 0 || config->nfreq <= 0) return MW_BATCH_ERR_INVALID_ARGUMENT;
 if (!isfinite(config->nu0_hz) || !isfinite(config->dlog10_nu) || config->nu0_hz <= 0.0)
  return MW_BATCH_ERR_INVALID_ARGUMENT;
 return MW_BATCH_OK;
}

void MWApproxBatchFillFrequencyGrid(const MWApproxBatchConfig *config, double *freq_hz)
{
 freq_hz[0]=config->nu0_hz;
 double dnu=pow(10.0, config->dlog10_nu);
 for (int i=1; i<config->nfreq; i++) freq_hz[i]=freq_hz[i-1]*dnu;
}

int MWApproxBatchRun(const MWApproxBatchConfig *config, const MWApproxBatchInputs *inputs, const MWApproxBatchOutputs *outputs)
{
 MWApproxBatchTimingResetImpl();
 int res=MWApproxBatchValidate(config, inputs, outputs);
 if (res) return res;

 MWApproxBatchFillFrequencyGrid(config, outputs->freq_hz);

 switch (config->backend)
 {
  case MW_BATCH_BACKEND_CPU:
   return MWApproxBatchRunCpu(config, inputs, outputs);
  case MW_BATCH_BACKEND_CUDA:
   return MWApproxBatchRunCuda(config, inputs, outputs);
  default:
   return MW_BATCH_ERR_UNSUPPORTED_BACKEND;
 }
}

int MWApproxBatchCudaAvailable(void)
{
 return MWApproxBatchCudaAvailableImpl();
}
