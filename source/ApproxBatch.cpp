#include <math.h>
#include <vector>
#include <string.h>
#include "ApproxBatchInternal.h"

namespace
{

MWApproxBatchTiming g_last_batch_timing{};

constexpr double MW_BATCH_SOLAR_AU_CM = 1.495978707e13;
constexpr double MW_BATCH_FLUX_SCALE = 1.0e-19;
constexpr double MW_BATCH_TAU_EXP_LIMIT = 700.0;

int MWApproxBatchValidateCommon(const MWApproxBatchConfig *config, const MWApproxBatchInputs *inputs)
{
 if (!config || !inputs) return MW_BATCH_ERR_INVALID_ARGUMENT;
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

int MWApproxBatchValidateRL(const MWApproxBatchConfig *config, const MWApproxBatchInputs *inputs, const MWApproxBatchRlOutputs *outputs, double d_sun_au)
{
 if (!outputs || !outputs->status || !outputs->freq_hz || !outputs->rl) return MW_BATCH_ERR_INVALID_ARGUMENT;
 if (!isfinite(d_sun_au) || d_sun_au <= 0.0) return MW_BATCH_ERR_INVALID_ARGUMENT;
 return MWApproxBatchValidateCommon(config, inputs);
}

size_t MWApproxBatchKernelIndex(int nfreq, int freq_index, int batch_index)
{
 return static_cast<size_t>(freq_index) + static_cast<size_t>(nfreq) * static_cast<size_t>(batch_index);
}

size_t MWApproxBatchRlIndex(int nfreq, int row_index, int freq_index, int batch_index)
{
 return static_cast<size_t>(row_index)
      + 7u * (static_cast<size_t>(freq_index) + static_cast<size_t>(nfreq) * static_cast<size_t>(batch_index));
}

void MWApproxBatchConvertLocalToRl(const MWApproxBatchConfig *config,
                                  const MWApproxBatchInputs *inputs,
                                  const double *freq_hz,
                                  const double *jx,
                                  const double *kx,
                                  const double *jo,
                                  const double *ko,
                                  double *rl,
                                  double d_sun_au)
{
 const size_t rl_size = static_cast<size_t>(7) * static_cast<size_t>(config->nfreq) * static_cast<size_t>(config->batch_size);
 memset(rl, 0, sizeof(double) * rl_size);

 const double distance_cm = d_sun_au * MW_BATCH_SOLAR_AU_CM;
 const double distance_scale = distance_cm * distance_cm * MW_BATCH_FLUX_SCALE;

 for (int batch_index=0; batch_index<config->batch_size; ++batch_index)
 {
  const double sang = inputs->area_cm2[batch_index] / distance_scale;
  const double depth_cm = inputs->depth_cm[batch_index];
  const bool x_is_left = inputs->theta_deg[batch_index] > 90.0;

  for (int freq_index=0; freq_index<config->nfreq; ++freq_index)
  {
   const size_t kernel_index = MWApproxBatchKernelIndex(config->nfreq, freq_index, batch_index);
   const size_t freq_row_index = MWApproxBatchRlIndex(config->nfreq, 0, freq_index, batch_index);
   const size_t left_row_index = MWApproxBatchRlIndex(config->nfreq, 5, freq_index, batch_index);
   const size_t right_row_index = MWApproxBatchRlIndex(config->nfreq, 6, freq_index, batch_index);

   const double tau_o = -ko[kernel_index] * depth_cm;
   const double tau_x = -kx[kernel_index] * depth_cm;
   const double e_o = (tau_o < MW_BATCH_TAU_EXP_LIMIT) ? exp(tau_o) : 0.0;
   const double e_x = (tau_x < MW_BATCH_TAU_EXP_LIMIT) ? exp(tau_x) : 0.0;

   double io = jo[kernel_index] * depth_cm;
   double ix = jx[kernel_index] * depth_cm;

   if (ko[kernel_index] != 0.0 && tau_o <= MW_BATCH_TAU_EXP_LIMIT)
   {
    double delta_o = 1.0 - e_o;
    if (delta_o == 0.0) delta_o = -tau_o;
    io = jo[kernel_index] / ko[kernel_index] * delta_o;
   }

   if (kx[kernel_index] != 0.0 && tau_x <= MW_BATCH_TAU_EXP_LIMIT)
   {
    double delta_x = 1.0 - e_x;
    if (delta_x == 0.0) delta_x = -tau_x;
    ix = jx[kernel_index] / kx[kernel_index] * delta_x;
   }

   rl[freq_row_index] = freq_hz[freq_index] / 1.0e9;
   if (x_is_left)
   {
    rl[left_row_index] = ix * sang;
    rl[right_row_index] = io * sang;
   }
   else
   {
    rl[left_row_index] = io * sang;
    rl[right_row_index] = ix * sang;
   }
  }
 }
}

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
 if (!outputs) return MW_BATCH_ERR_INVALID_ARGUMENT;
 if (!outputs->status || !outputs->freq_hz || !outputs->jx || !outputs->kx || !outputs->jo || !outputs->ko)
  return MW_BATCH_ERR_INVALID_ARGUMENT;
 return MWApproxBatchValidateCommon(config, inputs);
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

int MWApproxBatchRunRL(const MWApproxBatchConfig *config, const MWApproxBatchInputs *inputs, const MWApproxBatchRlOutputs *outputs, double d_sun_au)
{
 MWApproxBatchTimingResetImpl();
 int res=MWApproxBatchValidateRL(config, inputs, outputs, d_sun_au);
 if (res) return res;

 const size_t kernel_size = static_cast<size_t>(config->nfreq) * static_cast<size_t>(config->batch_size);
 std::vector<double> jx(kernel_size);
 std::vector<double> kx(kernel_size);
 std::vector<double> jo(kernel_size);
 std::vector<double> ko(kernel_size);

 MWApproxBatchOutputs local_outputs{};
 local_outputs.status=outputs->status;
 local_outputs.freq_hz=outputs->freq_hz;
 local_outputs.jx=jx.data();
 local_outputs.kx=kx.data();
 local_outputs.jo=jo.data();
 local_outputs.ko=ko.data();

 res=MWApproxBatchRun(config, inputs, &local_outputs);
 if (res) return res;

 MWApproxBatchConvertLocalToRl(config, inputs, outputs->freq_hz, jx.data(), kx.data(), jo.data(), ko.data(), outputs->rl, d_sun_au);
 return MW_BATCH_OK;
}

int MWApproxBatchCudaAvailable(void)
{
 return MWApproxBatchCudaAvailableImpl();
}
