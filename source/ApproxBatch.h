#pragma once

#define MW_BATCH_BACKEND_CPU 0
#define MW_BATCH_BACKEND_CUDA 1

#define MW_BATCH_PRECISION_FP64 0
#define MW_BATCH_PRECISION_FP32 1

#define MW_BATCH_OK 0
#define MW_BATCH_ERR_INVALID_ARGUMENT 1000
#define MW_BATCH_ERR_UNSUPPORTED_BACKEND 1001
#define MW_BATCH_ERR_CUDA_UNAVAILABLE 1002
#define MW_BATCH_ERR_CUDA_RUNTIME 1003

struct MWApproxBatchConfig
{
 int backend{};
 int precision{};
 int batch_size{};
 int nfreq{};
 int npoints{};
 int q_on{};
 double nu0_hz{};
 double dlog10_nu{};
};

struct MWApproxBatchInputs
{
 const double *area_cm2{};
 const double *depth_cm{};
 const double *bmag_g{};
 const double *temperature_k{};
 const double *thermal_density_cm3{};
 const double *nonthermal_density_cm3{};
 const double *delta{};
 const double *theta_deg{};
 const double *emin_mev{};
 const double *emax_mev{};
};

struct MWApproxBatchOutputs
{
 int *status{};
 double *freq_hz{};
 double *jx{};
 double *kx{};
 double *jo{};
 double *ko{};
};

struct MWApproxBatchTiming
{
 double total_seconds{};
 double setup_seconds{};
 double h2d_seconds{};
 double device_alloc_seconds{};
 double device_zero_seconds{};
 double backend_compute_seconds{};
 double sync_seconds{};
 double d2h_seconds{};
 double cleanup_seconds{};
};

int MWApproxBatchRun(const MWApproxBatchConfig *config, const MWApproxBatchInputs *inputs, const MWApproxBatchOutputs *outputs);
int MWApproxBatchCudaAvailable(void);
void MWApproxBatchTimingReset(void);
void MWApproxBatchTimingGet(MWApproxBatchTiming *timing);
