#include <algorithm>
#include <chrono>
#include <math.h>
#include <string.h>
#include <vector>
#include "ApproxBatchInternal.h"
#include "IDLinterface.h"
#include "MWmain.h"
#include "Std_DF.h"

namespace
{

int ValidateItem(const MWApproxBatchInputs *inputs, int i)
{
 if (!isfinite(inputs->area_cm2[i]) || inputs->area_cm2[i] <= 0.0) return MW_BATCH_ERR_INVALID_ARGUMENT;
 if (!isfinite(inputs->depth_cm[i]) || inputs->depth_cm[i] <= 0.0) return MW_BATCH_ERR_INVALID_ARGUMENT;
 if (!isfinite(inputs->bmag_g[i]) || inputs->bmag_g[i] <= 0.0) return MW_BATCH_ERR_INVALID_ARGUMENT;
 if (!isfinite(inputs->temperature_k[i]) || inputs->temperature_k[i] <= 0.0) return MW_BATCH_ERR_INVALID_ARGUMENT;
 if (!isfinite(inputs->thermal_density_cm3[i]) || inputs->thermal_density_cm3[i] <= 0.0) return MW_BATCH_ERR_INVALID_ARGUMENT;
 if (!isfinite(inputs->nonthermal_density_cm3[i]) || inputs->nonthermal_density_cm3[i] < 0.0) return MW_BATCH_ERR_INVALID_ARGUMENT;
 if (!isfinite(inputs->delta[i]) || inputs->delta[i] <= 1.0) return MW_BATCH_ERR_INVALID_ARGUMENT;
 if (!isfinite(inputs->theta_deg[i])) return MW_BATCH_ERR_INVALID_ARGUMENT;
 if (!isfinite(inputs->emin_mev[i]) || inputs->emin_mev[i] <= 0.0) return MW_BATCH_ERR_INVALID_ARGUMENT;
 if (!isfinite(inputs->emax_mev[i]) || inputs->emax_mev[i] <= inputs->emin_mev[i]) return MW_BATCH_ERR_INVALID_ARGUMENT;
 return MW_BATCH_OK;
}

void FillLparms(const MWApproxBatchConfig *config, int *lparms)
{
 memset(lparms, 0, sizeof(int)*11);
 lparms[i_Nz]=1;
 lparms[i_Nnu]=config->nfreq;
 lparms[i_Nnodes]=config->npoints;
 lparms[i_matchKey]=1;
 lparms[i_QoptKey]=config->q_on ? 0 : 1;
 lparms[i_arrKeyG]=1;
}

void FillRparms(const MWApproxBatchConfig *config, const MWApproxBatchInputs *inputs, int i, double *rparms)
{
 memset(rparms, 0, sizeof(double)*RpSize);
 rparms[i_S]=inputs->area_cm2[i];
 rparms[i_nu0]=config->nu0_hz;
 rparms[i_dnu]=config->dlog10_nu;
 rparms[i_nuCr]=0.0;
 rparms[i_nuWH]=0.0;
}

void FillParms(const MWApproxBatchInputs *inputs, int i, double *parms)
{
 memset(parms, 0, sizeof(double)*InSize);
 parms[i_dz]=inputs->depth_cm[i];
 parms[i_T0]=inputs->temperature_k[i];
 parms[i_n0]=inputs->thermal_density_cm3[i];
 parms[i_B]=inputs->bmag_g[i];
 parms[i_theta]=inputs->theta_deg[i];
 parms[i_EMflag]=6.0;
 parms[i_EId]=PLW;
 parms[i_nb]=inputs->nonthermal_density_cm3[i];
 parms[i_Emin]=inputs->emin_mev[i];
 parms[i_Emax]=inputs->emax_mev[i];
 parms[i_delta1]=inputs->delta[i];
 parms[i_muId]=ISO;
 parms[i_arrKeyL]=0.0;
}

}

int MWApproxBatchRunCpu(const MWApproxBatchConfig *config, const MWApproxBatchInputs *inputs, const MWApproxBatchOutputs *outputs)
{
 auto total_start=std::chrono::steady_clock::now();
 std::fill(outputs->status, outputs->status+config->batch_size, MW_BATCH_OK);
 std::fill(outputs->jx, outputs->jx+config->batch_size*config->nfreq, 0.0);
 std::fill(outputs->kx, outputs->kx+config->batch_size*config->nfreq, 0.0);
 std::fill(outputs->jo, outputs->jo+config->batch_size*config->nfreq, 0.0);
 std::fill(outputs->ko, outputs->ko+config->batch_size*config->nfreq, 0.0);

 std::vector<double> nu(config->nfreq);
 for (int i=0; i<config->nfreq; i++) nu[i]=outputs->freq_hz[i];
 double dummy[1]={0.0};

 for (int batch_index=0; batch_index<config->batch_size; batch_index++)
 {
  outputs->status[batch_index]=ValidateItem(inputs, batch_index);
  if (outputs->status[batch_index]) continue;

  int lparms[11];
  double rparms[RpSize];
  double parms[InSize];
  FillLparms(config, lparms);
  FillRparms(config, inputs, batch_index, rparms);
  FillParms(inputs, batch_index, parms);

  double *jx=outputs->jx+config->nfreq*batch_index;
  double *kx=outputs->kx+config->nfreq*batch_index;
  double *jo=outputs->jo+config->nfreq*batch_index;
  double *ko=outputs->ko+config->nfreq*batch_index;
  double ne_total=0.0;

  int res=FindLocalJK(nu.data(), lparms, rparms, parms, dummy, dummy, dummy, jx, jo, kx, ko, &ne_total);
  outputs->status[batch_index]=res;
  if (res)
  {
   std::fill(jx, jx+config->nfreq, 0.0);
   std::fill(kx, kx+config->nfreq, 0.0);
   std::fill(jo, jo+config->nfreq, 0.0);
   std::fill(ko, ko+config->nfreq, 0.0);
  }
 }

 MWApproxBatchTiming timing{};
 timing.total_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-total_start).count();
 timing.backend_compute_seconds=timing.total_seconds;
 MWApproxBatchTimingStore(&timing);
 return MW_BATCH_OK;
}
