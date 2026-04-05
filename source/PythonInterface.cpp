#include "IDLinterface.h"
#include "ApproxBatch.h"
#include "Messages.h"

#ifndef LINUX
extern "C" __declspec(dllexport) int pyGET_MW(int *Lparms, double *Rparms, double *Parms,
                                              double *E_arr, double *mu_arr, double *f_arr, double *RL)
#else
extern "C" double pyGET_MW(int *Lparms, double *Rparms, double *Parms,
                           double *E_arr, double *mu_arr, double *f_arr, double *RL)
#endif
{
 void *ARGV[7];
 ARGV[0]=(void*)Lparms;
 ARGV[1]=(void*)Rparms;
 ARGV[2]=(void*)Parms;
 ARGV[3]=(void*)E_arr;
 ARGV[4]=(void*)mu_arr;
 ARGV[5]=(void*)f_arr;
 ARGV[6]=(void*)RL; 

 return GET_MW(7, ARGV);
}

#ifndef LINUX
extern "C" __declspec(dllexport) int pyGET_MW_SLICE(int *Lparms_M, double *Rparms_M, double *Parms_M,
                                                    double *E_arr, double *mu_arr, double *f_arr_M, 
                                                    double *RL_M)
#else
extern "C" double pyGET_MW_SLICE(int *Lparms_M, double *Rparms_M, double *Parms_M,
                                 double *E_arr, double *mu_arr, double *f_arr_M, double *RL_M)
#endif
{
 void *ARGV[7];
 ARGV[0]=(void*)Lparms_M;
 ARGV[1]=(void*)Rparms_M;
 ARGV[2]=(void*)Parms_M;
 ARGV[3]=(void*)E_arr;
 ARGV[4]=(void*)mu_arr;
 ARGV[5]=(void*)f_arr_M;
 ARGV[6]=(void*)RL_M;

 return GET_MW_SLICE(7, ARGV);
}

#ifndef LINUX
extern "C" __declspec(dllexport) int pyMW_Approx_Batch(int backend, int precision, int batch_size, int nfreq, int npoints, int q_on,
                                                       double nu0_hz, double dlog10_nu,
                                                       double *area_cm2, double *depth_cm, double *bmag_g, double *temperature_k,
                                                       double *thermal_density_cm3, double *nonthermal_density_cm3, double *delta,
                                                       double *theta_deg, double *emin_mev, double *emax_mev,
                                                       int *status, double *freq_hz, double *jx, double *kx, double *jo, double *ko)
#else
extern "C" double pyMW_Approx_Batch(int backend, int precision, int batch_size, int nfreq, int npoints, int q_on,
                                    double nu0_hz, double dlog10_nu,
                                    double *area_cm2, double *depth_cm, double *bmag_g, double *temperature_k,
                                    double *thermal_density_cm3, double *nonthermal_density_cm3, double *delta,
                                    double *theta_deg, double *emin_mev, double *emax_mev,
                                    int *status, double *freq_hz, double *jx, double *kx, double *jo, double *ko)
#endif
{
 MWApproxBatchConfig config;
 config.backend=backend;
 config.precision=precision;
 config.batch_size=batch_size;
 config.nfreq=nfreq;
 config.npoints=npoints;
 config.q_on=q_on;
 config.nu0_hz=nu0_hz;
 config.dlog10_nu=dlog10_nu;

 MWApproxBatchInputs inputs;
 inputs.area_cm2=area_cm2;
 inputs.depth_cm=depth_cm;
 inputs.bmag_g=bmag_g;
 inputs.temperature_k=temperature_k;
 inputs.thermal_density_cm3=thermal_density_cm3;
 inputs.nonthermal_density_cm3=nonthermal_density_cm3;
 inputs.delta=delta;
 inputs.theta_deg=theta_deg;
 inputs.emin_mev=emin_mev;
 inputs.emax_mev=emax_mev;

 MWApproxBatchOutputs outputs;
 outputs.status=status;
 outputs.freq_hz=freq_hz;
 outputs.jx=jx;
 outputs.kx=kx;
 outputs.jo=jo;
 outputs.ko=ko;

 return MWApproxBatchRun(&config, &inputs, &outputs);
}

#ifndef LINUX
extern "C" __declspec(dllexport) int pyMW_Approx_Batch_CudaAvailable(void)
#else
extern "C" double pyMW_Approx_Batch_CudaAvailable(void)
#endif
{
 return MWApproxBatchCudaAvailable();
}

#ifndef LINUX
extern "C" __declspec(dllexport) void pyMW_Approx_Batch_TimingReset(void)
#else
extern "C" void pyMW_Approx_Batch_TimingReset(void)
#endif
{
 MWApproxBatchTimingReset();
}

#ifndef LINUX
extern "C" __declspec(dllexport) void pyMW_Approx_Batch_TimingGet(MWApproxBatchTiming *timing)
#else
extern "C" void pyMW_Approx_Batch_TimingGet(MWApproxBatchTiming *timing)
#endif
{
 MWApproxBatchTimingGet(timing);
}
