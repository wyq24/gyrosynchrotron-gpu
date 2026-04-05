#include <math.h>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>
#include "ApproxBatchInternal.h"
#include "Plasma.h"

namespace
{

const int kMaxSecantIters=20;
const int kMaxBrentIters=100;

template <typename Real>
__host__ __device__ inline Real sqr_r(Real x)
{
 return x*x;
}

template <typename Real>
__host__ __device__ inline Real sign_r(Real x)
{
 return (x < Real(0)) ? Real(-1) : Real(1);
}

template <typename Real>
__host__ __device__ inline Real invalid_r()
{
 return Real(HUGE_VAL);
}

template <typename Real>
__host__ __device__ inline Real pi_r()
{
 return Real(3.14159265358979323846264338327950288);
}

__host__ __device__ inline double legacy_absorption_sentinel_d()
{
 return 1e100;
}

template <typename Real>
struct WaveState
{
 int valid{};
 Real nu{}, nu_p{}, nu_B{}, theta{};
 Real ct{}, st{};
 Real y{}, N{}, N_z{}, T{}, L{};
};

template <typename Real>
__host__ __device__ WaveState<Real> MakeWaveState(Real nu, Real theta, int sigma, Real nu_p, Real nu_B)
{
 WaveState<Real> w;
 w.nu=nu;
 w.theta=theta;
 w.nu_p=nu_p;
 w.nu_B=nu_B;

 Real nu_c=(sigma==-1) ? nu_B/Real(2)+sqrt(sqr_r(nu_p)+sqr_r(nu_B)/Real(4)) : nu_p;
 if (nu<=nu_c)
 {
  w.valid=0;
  return w;
 }

 w.valid=1;
 w.ct=cos(theta);
 w.st=sin(theta);

 if (w.st<Real(0))
 {
  w.st=-w.st;
  w.ct=-w.ct;
 }

 if (fabs(w.ct)<Real(cst_min))
 {
  w.ct=Real(cst_min)*sign_r(w.ct);
  w.st=sqrt(Real(1)-sqr_r(w.ct))*sign_r(w.st);
 }
 if (fabs(w.st)<Real(cst_min))
 {
  w.st=Real(cst_min)*sign_r(w.st);
  w.ct=sqrt(Real(1)-sqr_r(w.st))*sign_r(w.ct);
 }

 w.y=nu/nu_B;

 Real u=sqr_r(nu_B/nu);
 Real v=sqr_r(nu_p/nu);
 Real delta=sqrt(sqr_r(u*sqr_r(w.st))+Real(4)*u*sqr_r((Real(1)-v)*w.ct));
 w.N=sqrt(Real(1)-Real(2)*v*(Real(1)-v)/(Real(2)*(Real(1)-v)-u*sqr_r(w.st)+Real(sigma)*delta));
 w.N_z=w.N*w.ct;
 w.T=Real(2)*sqrt(u)*(Real(1)-v)*w.ct/(u*sqr_r(w.st)-Real(sigma)*delta);
 w.L=(v*sqrt(u)*w.st+w.T*u*v*w.st*w.ct)/(Real(1)-u-v+u*v*sqr_r(w.ct));
 w.valid=isfinite(w.N) ? 1 : 0;
 return w;
}

template <typename Real>
struct ApproxEvalState
{
 WaveState<Real> w;
 Real norm_a{};
 Real delta{};
 int q_on{};
 int qcorr{};
 int mflag{};
 Real mu_list[2]{};
 Real lnq1_list[2]{};
 Real e_loc{};
 Real beta_loc{};
 Real g_loc{};
 Real p_loc{};

 __host__ __device__ Real H1(Real mu)
 {
  Real nbct=w.N*beta_loc*w.ct;
  Real nbmct1=Real(1)-nbct*mu;
  Real sa2=Real(1)-sqr_r(mu);
  if (sa2<=Real(0) || nbmct1==Real(0)) return invalid_r<Real>();

  Real sa=sqrt(sa2);
  Real x=w.N*beta_loc*w.st*sa/nbmct1;
  if (!(x>Real(0)) || fabs(x)>=Real(1)) return invalid_r<Real>();

  Real s1mx2=sqrt(Real(1)-sqr_r(x));
  Real lnz=s1mx2+log(x/(Real(1)+s1mx2));

  Real lnq1=Real(0);
  if (qcorr)
  {
   Real s1mx2_3=s1mx2*s1mx2*s1mx2;
   Real s=g_loc*w.y*nbmct1;
   if (s<=Real(0)) return invalid_r<Real>();
   Real a6=s1mx2_3+Real(0.503297)/s;
   Real b16=s1mx2_3+Real(1.193000)/s;
   Real b2=Real(1)-Real(0.2)*pow(s, Real(-2.0/3.0));
   Real ab=pow(a6*b16, Real(1.0/6.0))*b2;
   Real xi=Real(3)*sqr_r(x)*s1mx2*(w.N_z*beta_loc-mu)/sa2;
   Real eta=w.N_z*beta_loc/s;
   Real lambda=g_loc*w.y/(Real(6)*s);
   Real a_1a=lambda*(Real(0.503297)*eta-xi)/a6;
   Real b_1b=lambda*(Real(1.193000)*eta-xi)/b16+Real(4)*lambda*beta_loc*w.N_z*(b2-Real(1))/b2;
   Real denom=w.T*(w.ct-w.N*beta_loc*mu)+w.L*w.st+ab*nbmct1;
   if (denom==Real(0)) return invalid_r<Real>();
   lnq1=Real(2)*(ab*(a_1a+b_1b)*nbmct1-ab*w.N_z*beta_loc-w.T*w.N*beta_loc)/denom-Real(2)*a_1a+w.N_z*beta_loc/nbmct1;

   mu_list[mflag]=mu;
   lnq1_list[mflag]=lnq1;
   mflag^=1;
  }

  return Real(2)*g_loc*w.y*((nbct-mu)/sa2*s1mx2-nbct*lnz)+lnq1;
 }
};

template <typename Real>
struct MuSolveFunction
{
 ApproxEvalState<Real> *state{};

 __host__ __device__ Real Eval(Real mu)
 {
  return state->H1(mu);
 }
};

template <typename Real, typename Func>
__host__ __device__ Real SecantRootR(Func *f, Real x1, Real x2, Real eps)
{
 Real fl=f->Eval(x1);
 Real fx=f->Eval(x2);
 if (!isfinite(fl) || !isfinite(fx)) return invalid_r<Real>();

 Real dx, swap, xl, rts;
 if (fabs(fl)<fabs(fx))
 {
  rts=x1;
  xl=x2;
  swap=fl;
  fl=fx;
  fx=swap;
 }
 else
 {
  xl=x1;
  rts=x2;
 }

 int j=0;
 do
 {
  if (fx==fl) return invalid_r<Real>();
  dx=(xl-rts)*fx/(fx-fl);
  xl=rts;
  fl=fx;
  rts+=dx;
  fx=f->Eval(rts);
  j++;
 } while (fabs(dx)>eps && fx!=Real(0) && j<kMaxSecantIters);

 return (j<kMaxSecantIters && isfinite(fx)) ? rts : invalid_r<Real>();
}

template <typename Real, typename Func>
__host__ __device__ Real BrentRootR(Func *f, Real x1, Real x2, Real tol)
{
 Real a=x1, b=x2, c_bracket=x2, d=Real(0), step_prev=Real(0);
 Real fa=f->Eval(a), fb=f->Eval(b), fc, p, q, r, s, tol1, xm;
 if (!isfinite(fa) || !isfinite(fb)) return invalid_r<Real>();
 if (fa*fb>Real(0)) return invalid_r<Real>();

 fc=fb;
 for (int iter=1; iter<=kMaxBrentIters; iter++)
 {
  if ((fb>Real(0) && fc>Real(0)) || (fb<Real(0) && fc<Real(0)))
  {
   c_bracket=a;
   fc=fa;
   step_prev=d=b-a;
  }
  if (fabs(fc)<fabs(fb))
  {
   a=b;
   fa=fb;
   b=c_bracket;
   fb=fc;
   // Intentionally preserve the legacy CPU BrentRoot update order so the
   // CUDA port follows the same root-selection path on the supported narrow
   // batch workload.
   c_bracket=a;
   fc=fa;
  }
  tol1=Real(0.5)*tol;
  xm=Real(0.5)*(c_bracket-b);
  if (fabs(xm)<=tol1 || fb==Real(0)) return b;
  if (fabs(step_prev)>=tol1 && fabs(fa)>fabs(fb))
  {
   s=fb/fa;
   if (a==c_bracket)
   {
    p=Real(2)*xm*s;
    q=Real(1)-s;
   }
   else
   {
    q=fa/fc;
    r=fb/fc;
    p=s*(Real(2)*xm*q*(q-r)-(b-a)*(r-Real(1)));
    q=(q-Real(1))*(r-Real(1))*(s-Real(1));
   }
   if (p>Real(0)) q=-q;
   p=fabs(p);
   Real min1=Real(3)*xm*q-fabs(tol1*q);
   Real min2=fabs(step_prev*q);
   if (Real(2)*p<((min1<min2) ? min1 : min2))
   {
    step_prev=d;
    d=p/q;
   }
   else
   {
    d=xm;
    step_prev=d;
   }
  }
  else
  {
   d=xm;
   step_prev=d;
  }

  a=b;
  fa=fb;
  if (fabs(d)>tol1) b+=d;
  else b+=((xm>=Real(0)) ? tol1 : -tol1);
  fb=f->Eval(b);
  if (!isfinite(fb)) return invalid_r<Real>();
 }

 return invalid_r<Real>();
}

template <typename Real>
__host__ __device__ void FindMu0(ApproxEvalState<Real> *state, Real *mu0, Real *lnq2)
{
 MuSolveFunction<Real> f;
 f.state=state;

 *lnq2=Real(0);
 state->qcorr=0;
 *mu0=BrentRootR<Real>(&f, Real(-1)+Real(1e-5), Real(1)-Real(1e-5), Real(1e-3));
 if (!isfinite(*mu0) || !state->q_on) return;

 state->mflag=0;
 int qfound=0;
 state->qcorr=1;

 Real mu1=SecantRootR<Real>(&f, *mu0, *mu0-Real(1e-4)*sign_r(*mu0), Real(1e-3));
 if (isfinite(mu1) && fabs(mu1)<Real(1))
 {
  *mu0=mu1;
  qfound=1;
 }
 else
 {
  mu1=BrentRootR<Real>(&f, Real(-1)+Real(1e-5), *mu0, Real(1e-3));
  if (isfinite(mu1))
  {
   *mu0=mu1;
   qfound=1;
  }
  else
  {
   mu1=BrentRootR<Real>(&f, *mu0, Real(1)-Real(1e-5), Real(1e-3));
   if (isfinite(mu1))
   {
    *mu0=mu1;
    qfound=1;
   }
  }
 }

 Real dmu=state->mu_list[1]-state->mu_list[0];
 if (qfound && dmu!=Real(0)) *lnq2=(state->lnq1_list[1]-state->lnq1_list[0])/dmu;
}

template <typename Real>
__host__ __device__ void EvaluateEnergy(ApproxEvalState<Real> *state, Real e_erg, Real *j_val, Real *k_val)
{
 *j_val=Real(0);
 *k_val=Real(0);
 if (e_erg==Real(0)) return;

 state->e_loc=e_erg;
 state->g_loc=e_erg/Real(mc2)+Real(1);
 state->beta_loc=sqrt(sqr_r(state->g_loc)-Real(1))/state->g_loc;
 state->p_loc=state->beta_loc*state->g_loc*Real(mc);

 Real mu0, lnq2;
 FindMu0(state, &mu0, &lnq2);
 if (!isfinite(mu0)) return;

 if (!(state->norm_a>Real(0))) return;
 Real log_f=log(Real(0.5)*state->norm_a)-state->delta*log(e_erg);
 Real f=exp(log_f);
 if (!isfinite(f)) return;
 Real df_de=(-state->delta/e_erg)*f;
 Real nbct=state->w.N*state->beta_loc*state->w.ct;
 Real nbctm=nbct-mu0;
 Real nbmct1=Real(1)-nbct*mu0;
 Real sa2=Real(1)-sqr_r(mu0);
 if (sa2<=Real(0) || nbmct1==Real(0)) return;
 Real sa=sqrt(sa2);
 Real x=state->w.N*state->beta_loc*state->w.st*sa/nbmct1;
 if (!(x>Real(0)) || fabs(x)>=Real(1)) return;

 Real s1mx2=sqrt(Real(1)-sqr_r(x));
 Real s1mx2_3=s1mx2*s1mx2*s1mx2;
 Real s=state->g_loc*state->w.y*nbmct1;
 if (s<=Real(0)) return;
 Real a=pow(s1mx2_3+Real(0.503297)/s, Real(1.0/6.0));
 Real b=pow(s1mx2_3+Real(1.193000)/s, Real(1.0/6.0))*(Real(1)-Real(0.2)*pow(s, Real(-2.0/3.0)));

 Real q=sqr_r(state->w.T*(state->w.ct-state->w.N*state->beta_loc*mu0)+state->w.L*state->w.st+a*b*nbmct1)/
        sqr_r(a)/nbmct1;

 Real lnz=s1mx2+log(x/(Real(1)+s1mx2));
 Real z=exp(Real(2)*s*lnz);
 Real h2=-Real(2)*state->g_loc*state->w.y*s1mx2/sa2*
         (Real(1)+sqr_r(state->w.N*state->beta_loc*state->w.st*nbctm/nbmct1)/nbmct1/(Real(1)-sqr_r(x))-
          Real(2)*mu0*nbctm/sa2+nbct*nbctm/nbmct1)+lnq2;
 Real lpfactor=sqrt(-Real(2)*pi_r<Real>()/h2);
 if (!isfinite(lpfactor)) lpfactor=Real(0);

 q*=z*lpfactor;
 Real r=df_de-(Real(1)+sqr_r(state->beta_loc))/(Real(c)*state->p_loc*state->beta_loc)*f;

 *j_val=q*f*e_erg;
 *k_val=q*r*e_erg;
}

template <typename Real>
__global__ void ApproxBatchKernel(int batch_size, int nfreq, int npoints, int q_on, double nu0_hz, double dlog10_nu,
                                  const int *valid_mask,
                                  const double *bmag_g, const double *thermal_density_cm3, const double *nonthermal_density_cm3,
                                  const double *temperature_k, const double *delta, const double *theta_deg,
                                  const double *emin_mev, const double *emax_mev,
                                  double *jx, double *kx, double *jo, double *ko)
{
 int idx=blockIdx.x*blockDim.x+threadIdx.x;
 int total=batch_size*nfreq*2;
 if (idx>=total) return;

 int spectrum_index=idx/(nfreq*2);
 if (!valid_mask[spectrum_index]) return;

 int rem=idx%(nfreq*2);
 int freq_index=rem/2;
 int pol_index=rem%2;
 int sigma=(pol_index==0) ? -1 : 1;

 Real e1=Real(emin_mev[spectrum_index])*Real(eV)*Real(1e6);
 Real e2=Real(emax_mev[spectrum_index])*Real(eV)*Real(1e6);
 Real delta_loc=Real(delta[spectrum_index]);
 Real nb=Real(nonthermal_density_cm3[spectrum_index]);
 Real delta_minus_one=delta_loc-Real(1);
 if (delta_minus_one<=Real(0)) return;
 Real e_ratio=e1/e2;
 if (!(e_ratio>Real(0)) || !(e_ratio<Real(1))) return;
 Real ratio_term=pow(e_ratio, delta_minus_one);
 Real norm_scale=Real(1)-ratio_term;
 if (!(norm_scale>Real(0))) return;
 Real norm_a=nb/(Real(2)*pi_r<Real>())*delta_minus_one*pow(e1, delta_minus_one)/norm_scale;

 Real ne_total=Real(thermal_density_cm3[spectrum_index])+nb;
 Real nu_p=Real(e)*sqrt(ne_total/Real(me)/pi_r<Real>());
 Real nu_b=Real(e)*Real(bmag_g[spectrum_index])/Real(me)/Real(c)/(Real(2)*pi_r<Real>());
 Real theta=Real(theta_deg[spectrum_index])*pi_r<Real>()/Real(180);
 Real nu=Real(nu0_hz)*pow(Real(10), Real(dlog10_nu)*Real(freq_index));

 WaveState<Real> w=MakeWaveState(nu, theta, sigma, nu_p, nu_b);
 int invalid_wave=!w.valid;
 Real j_res=Real(0);
 Real k_res=Real(0);

 if (w.valid && nu_b>Real(0))
 {
  ApproxEvalState<Real> state;
  state.w=w;
  state.norm_a=norm_a;
  state.delta=delta_loc;
  state.q_on=q_on;

  Real t1=log(e1);
  Real t2=log(e2);
  Real dt=(t2-t1)/Real(npoints);

  Real sum_j=Real(0);
  Real sum_k=Real(0);
  for (int i=0; i<=npoints; i++)
  {
   Real t=t1+dt*Real(i);
   Real e_erg=exp(t);
   Real val_j, val_k;
   EvaluateEnergy(&state, e_erg, &val_j, &val_k);
   if (i==0 || i==npoints)
   {
    val_j/=Real(2);
    val_k/=Real(2);
   }
   sum_j+=val_j;
   sum_k+=val_k;
  }

  j_res=sum_j*dt*(Real(2)*pi_r<Real>()*sqr_r(Real(e))*w.nu/Real(c)/w.N/(Real(1)+sqr_r(w.T))/sqr_r(w.st));
  k_res=sum_k*dt*(-Real(2)*pi_r<Real>()*sqr_r(Real(e))*Real(c)/sqr_r(w.N)/w.N/w.nu/(Real(1)+sqr_r(w.T))/sqr_r(w.st));
 }

 int out_index=freq_index+nfreq*spectrum_index;
 double k_store=invalid_wave ? legacy_absorption_sentinel_d() : double(k_res);
 if (sigma==-1)
 {
  jx[out_index]=double(j_res);
  kx[out_index]=k_store;
 }
 else
 {
  jo[out_index]=double(j_res);
  ko[out_index]=k_store;
 }
}

int ValidateCudaItem(const MWApproxBatchInputs *inputs, int i)
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

template <typename T>
int CopyArrayToDevice(const T *host_ptr, int count, T **device_ptr)
{
 cudaError_t err=cudaMalloc((void**)device_ptr, sizeof(T)*count);
 if (err!=cudaSuccess) return MW_BATCH_ERR_CUDA_RUNTIME;
 err=cudaMemcpy(*device_ptr, host_ptr, sizeof(T)*count, cudaMemcpyHostToDevice);
 if (err!=cudaSuccess) return MW_BATCH_ERR_CUDA_RUNTIME;
 return MW_BATCH_OK;
}

}

int MWApproxBatchRunCuda(const MWApproxBatchConfig *config, const MWApproxBatchInputs *inputs, const MWApproxBatchOutputs *outputs)
{
 auto total_start=std::chrono::steady_clock::now();
 MWApproxBatchTiming timing{};
 std::fill(outputs->status, outputs->status+config->batch_size, MW_BATCH_OK);
 std::fill(outputs->jx, outputs->jx+config->batch_size*config->nfreq, 0.0);
 std::fill(outputs->kx, outputs->kx+config->batch_size*config->nfreq, 0.0);
 std::fill(outputs->jo, outputs->jo+config->batch_size*config->nfreq, 0.0);
 std::fill(outputs->ko, outputs->ko+config->batch_size*config->nfreq, 0.0);

 if (config->npoints <= 0) return MW_BATCH_ERR_INVALID_ARGUMENT;

 std::vector<int> valid_mask(config->batch_size, 1);
 for (int i=0; i<config->batch_size; i++)
 {
  outputs->status[i]=ValidateCudaItem(inputs, i);
  if (outputs->status[i]) valid_mask[i]=0;
 }

 double *d_bmag=0;
 double *d_temp=0;
 double *d_n0=0;
 double *d_nb=0;
 double *d_delta=0;
 double *d_theta=0;
 double *d_emin=0;
 double *d_emax=0;
 double *d_jx=0;
 double *d_kx=0;
 double *d_jo=0;
 double *d_ko=0;
 int *d_valid=0;
 int res=MW_BATCH_OK;
 int total_threads=config->batch_size*config->nfreq*2;
 int block_size=128;
 int grid_size=(total_threads+block_size-1)/block_size;
 cudaEvent_t kernel_start_event=0;
 cudaEvent_t kernel_stop_event=0;
 std::chrono::steady_clock::time_point h2d_start;
 std::chrono::steady_clock::time_point alloc_start;
 std::chrono::steady_clock::time_point zero_start;
 std::chrono::steady_clock::time_point sync_start;
 std::chrono::steady_clock::time_point d2h_start;
 float kernel_ms=0.0f;

 timing.setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-total_start).count();

 cudaError_t err=cudaSetDevice(0);
 if (err!=cudaSuccess) return MW_BATCH_ERR_CUDA_RUNTIME;

 h2d_start=std::chrono::steady_clock::now();
 res=CopyArrayToDevice(inputs->bmag_g, config->batch_size, &d_bmag); if (res) goto cleanup;
 res=CopyArrayToDevice(inputs->temperature_k, config->batch_size, &d_temp); if (res) goto cleanup;
 res=CopyArrayToDevice(inputs->thermal_density_cm3, config->batch_size, &d_n0); if (res) goto cleanup;
 res=CopyArrayToDevice(inputs->nonthermal_density_cm3, config->batch_size, &d_nb); if (res) goto cleanup;
 res=CopyArrayToDevice(inputs->delta, config->batch_size, &d_delta); if (res) goto cleanup;
 res=CopyArrayToDevice(inputs->theta_deg, config->batch_size, &d_theta); if (res) goto cleanup;
 res=CopyArrayToDevice(inputs->emin_mev, config->batch_size, &d_emin); if (res) goto cleanup;
 res=CopyArrayToDevice(inputs->emax_mev, config->batch_size, &d_emax); if (res) goto cleanup;
 res=CopyArrayToDevice(valid_mask.data(), config->batch_size, &d_valid); if (res) goto cleanup;
 timing.h2d_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-h2d_start).count();

 alloc_start=std::chrono::steady_clock::now();
 err=cudaMalloc((void**)&d_jx, sizeof(double)*config->batch_size*config->nfreq); if (err!=cudaSuccess) { res=MW_BATCH_ERR_CUDA_RUNTIME; goto cleanup; }
 err=cudaMalloc((void**)&d_kx, sizeof(double)*config->batch_size*config->nfreq); if (err!=cudaSuccess) { res=MW_BATCH_ERR_CUDA_RUNTIME; goto cleanup; }
 err=cudaMalloc((void**)&d_jo, sizeof(double)*config->batch_size*config->nfreq); if (err!=cudaSuccess) { res=MW_BATCH_ERR_CUDA_RUNTIME; goto cleanup; }
 err=cudaMalloc((void**)&d_ko, sizeof(double)*config->batch_size*config->nfreq); if (err!=cudaSuccess) { res=MW_BATCH_ERR_CUDA_RUNTIME; goto cleanup; }
 timing.device_alloc_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-alloc_start).count();
 zero_start=std::chrono::steady_clock::now();
 err=cudaMemset(d_jx, 0, sizeof(double)*config->batch_size*config->nfreq); if (err!=cudaSuccess) { res=MW_BATCH_ERR_CUDA_RUNTIME; goto cleanup; }
 err=cudaMemset(d_kx, 0, sizeof(double)*config->batch_size*config->nfreq); if (err!=cudaSuccess) { res=MW_BATCH_ERR_CUDA_RUNTIME; goto cleanup; }
 err=cudaMemset(d_jo, 0, sizeof(double)*config->batch_size*config->nfreq); if (err!=cudaSuccess) { res=MW_BATCH_ERR_CUDA_RUNTIME; goto cleanup; }
 err=cudaMemset(d_ko, 0, sizeof(double)*config->batch_size*config->nfreq); if (err!=cudaSuccess) { res=MW_BATCH_ERR_CUDA_RUNTIME; goto cleanup; }
 timing.device_zero_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-zero_start).count();
 err=cudaEventCreate(&kernel_start_event); if (err!=cudaSuccess) { res=MW_BATCH_ERR_CUDA_RUNTIME; goto cleanup; }
 err=cudaEventCreate(&kernel_stop_event); if (err!=cudaSuccess) { res=MW_BATCH_ERR_CUDA_RUNTIME; goto cleanup; }
 err=cudaEventRecord(kernel_start_event, 0); if (err!=cudaSuccess) { res=MW_BATCH_ERR_CUDA_RUNTIME; goto cleanup; }

 if (config->precision==MW_BATCH_PRECISION_FP32)
 {
  ApproxBatchKernel<float><<<grid_size, block_size>>>(
   config->batch_size, config->nfreq, config->npoints, config->q_on, config->nu0_hz, config->dlog10_nu,
   d_valid, d_bmag, d_n0, d_nb, d_temp, d_delta, d_theta, d_emin, d_emax, d_jx, d_kx, d_jo, d_ko);
 }
 else
 {
  ApproxBatchKernel<double><<<grid_size, block_size>>>(
   config->batch_size, config->nfreq, config->npoints, config->q_on, config->nu0_hz, config->dlog10_nu,
   d_valid, d_bmag, d_n0, d_nb, d_temp, d_delta, d_theta, d_emin, d_emax, d_jx, d_kx, d_jo, d_ko);
 }

 err=cudaEventRecord(kernel_stop_event, 0);
 if (err!=cudaSuccess) { res=MW_BATCH_ERR_CUDA_RUNTIME; goto cleanup; }
 err=cudaGetLastError();
 if (err!=cudaSuccess) { res=MW_BATCH_ERR_CUDA_RUNTIME; goto cleanup; }
 sync_start=std::chrono::steady_clock::now();
 err=cudaDeviceSynchronize();
 if (err!=cudaSuccess) { res=MW_BATCH_ERR_CUDA_RUNTIME; goto cleanup; }
 timing.sync_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-sync_start).count();
 err=cudaEventElapsedTime(&kernel_ms, kernel_start_event, kernel_stop_event);
 if (err!=cudaSuccess) { res=MW_BATCH_ERR_CUDA_RUNTIME; goto cleanup; }
 timing.backend_compute_seconds=double(kernel_ms)*1.0e-3;

 d2h_start=std::chrono::steady_clock::now();
 err=cudaMemcpy(outputs->jx, d_jx, sizeof(double)*config->batch_size*config->nfreq, cudaMemcpyDeviceToHost);
 if (err!=cudaSuccess) { res=MW_BATCH_ERR_CUDA_RUNTIME; goto cleanup; }
 err=cudaMemcpy(outputs->kx, d_kx, sizeof(double)*config->batch_size*config->nfreq, cudaMemcpyDeviceToHost);
 if (err!=cudaSuccess) { res=MW_BATCH_ERR_CUDA_RUNTIME; goto cleanup; }
 err=cudaMemcpy(outputs->jo, d_jo, sizeof(double)*config->batch_size*config->nfreq, cudaMemcpyDeviceToHost);
 if (err!=cudaSuccess) { res=MW_BATCH_ERR_CUDA_RUNTIME; goto cleanup; }
 err=cudaMemcpy(outputs->ko, d_ko, sizeof(double)*config->batch_size*config->nfreq, cudaMemcpyDeviceToHost);
 if (err!=cudaSuccess) { res=MW_BATCH_ERR_CUDA_RUNTIME; goto cleanup; }
 timing.d2h_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-d2h_start).count();

cleanup:
 auto cleanup_start=std::chrono::steady_clock::now();
 if (d_bmag) cudaFree(d_bmag);
 if (d_temp) cudaFree(d_temp);
 if (d_n0) cudaFree(d_n0);
 if (d_nb) cudaFree(d_nb);
 if (d_delta) cudaFree(d_delta);
 if (d_theta) cudaFree(d_theta);
 if (d_emin) cudaFree(d_emin);
 if (d_emax) cudaFree(d_emax);
 if (d_valid) cudaFree(d_valid);
 if (d_jx) cudaFree(d_jx);
 if (d_kx) cudaFree(d_kx);
 if (d_jo) cudaFree(d_jo);
 if (d_ko) cudaFree(d_ko);
 if (kernel_start_event) cudaEventDestroy(kernel_start_event);
 if (kernel_stop_event) cudaEventDestroy(kernel_stop_event);
 timing.cleanup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-cleanup_start).count();
 timing.total_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-total_start).count();
 MWApproxBatchTimingStore(&timing);
 return res;
}

int MWApproxBatchCudaAvailableImpl(void)
{
 int count=0;
 cudaError_t err=cudaGetDeviceCount(&count);
 if (err!=cudaSuccess) return 0;
 return count>0 ? 1 : 0;
}
