#pragma once

#include <stddef.h>
#include <stdint.h>

#include <limits>
#include <sstream>
#include <vector>

#include "ApproxBatch.h"
#include "xla/ffi/api/ffi.h"

namespace mw_xla
{

namespace ffi = xla::ffi;

constexpr int MW_APPROX_PARAM_DIM = 10;
constexpr int MW_APPROX_RL_ROWS = 7;

template <typename Dims>
inline int64_t Product(const Dims &dims, size_t limit)
{
 int64_t value = 1;
 for (size_t i = 0; i < limit; ++i) value *= dims[i];
 return value;
}

inline ffi::Error ValidateBuffers(const ffi::Buffer<ffi::DataType::F64> &params,
                                  ffi::ResultBuffer<ffi::DataType::F64> &rl,
                                  ffi::ResultBufferR1<ffi::DataType::F64> &freq,
                                  int64_t *flat_batch_size,
                                  int64_t *nfreq)
{
 if (!flat_batch_size || !nfreq)
  return ffi::Error::Internal("flat_batch_size and nfreq outputs must be non-null");

 const auto param_dims = params.dimensions();
 const auto rl_dims = rl->dimensions();
 const auto freq_dims = freq->dimensions();

 if (param_dims.size() < 1)
  return ffi::Error::InvalidArgument("params must have at least one dimension and end with size 10");
 if (param_dims[param_dims.size() - 1] != MW_APPROX_PARAM_DIM)
  return ffi::Error::InvalidArgument("params last dimension must have size 10");

 if (rl_dims.size() != param_dims.size() + 1)
  return ffi::Error::InvalidArgument("rl output rank must equal params rank + 1");
 if (rl_dims[rl_dims.size() - 1] != MW_APPROX_RL_ROWS)
  return ffi::Error::InvalidArgument("rl output last dimension must have size 7");

 for (size_t i = 0; i + 1 < param_dims.size(); ++i)
 {
  if (rl_dims[i] != param_dims[i])
   return ffi::Error::InvalidArgument("rl prefix dimensions must match params prefix dimensions");
 }

 *nfreq = freq_dims[0];
 if (rl_dims[rl_dims.size() - 2] != *nfreq)
  return ffi::Error::InvalidArgument("rl frequency dimension must match freq output length");

 *flat_batch_size = Product(param_dims, param_dims.size() - 1);
 if (*flat_batch_size <= 0)
  return ffi::Error::InvalidArgument("params must contain at least one spectrum");

 return ffi::Error::Success();
}

inline ffi::Error RunRlFromPackedParamsHost(int backend,
                                            int32_t npoints,
                                            int32_t q_on,
                                            double nu0_hz,
                                            double dlog10_nu,
                                            double d_sun_au,
                                            int batch_size,
                                            int nfreq,
                                            const double *params_row_major,
                                            double *rl_out,
                                            double *freq_out)
{
 if (!params_row_major || !rl_out || !freq_out)
  return ffi::Error::InvalidArgument("params_row_major, rl_out, and freq_out must be non-null");

 auto param_at = [&](int batch_index, int column_index) -> double {
  return params_row_major[
      static_cast<size_t>(batch_index) * MW_APPROX_PARAM_DIM + static_cast<size_t>(column_index)];
 };

 std::vector<double> area_cm2(batch_size);
 std::vector<double> depth_cm(batch_size);
 std::vector<double> bmag_g(batch_size);
 std::vector<double> temperature_k(batch_size);
 std::vector<double> thermal_density_cm3(batch_size);
 std::vector<double> nonthermal_density_cm3(batch_size);
 std::vector<double> delta(batch_size);
 std::vector<double> theta_deg(batch_size);
 std::vector<double> emin_mev(batch_size);
 std::vector<double> emax_mev(batch_size);
 std::vector<int> status(batch_size, 0);

 for (int i = 0; i < batch_size; ++i)
 {
  area_cm2[i] = param_at(i, 0);
  depth_cm[i] = param_at(i, 1);
  bmag_g[i] = param_at(i, 2);
  temperature_k[i] = param_at(i, 3);
  thermal_density_cm3[i] = param_at(i, 4);
  nonthermal_density_cm3[i] = param_at(i, 5);
  delta[i] = param_at(i, 6);
  theta_deg[i] = param_at(i, 7);
  emin_mev[i] = param_at(i, 8);
  emax_mev[i] = param_at(i, 9);
 }

 MWApproxBatchConfig config{};
 config.backend = backend;
 config.precision = MW_BATCH_PRECISION_FP64;
 config.batch_size = batch_size;
 config.nfreq = nfreq;
 config.npoints = static_cast<int>(npoints);
 config.q_on = static_cast<int>(q_on);
 config.nu0_hz = nu0_hz;
 config.dlog10_nu = dlog10_nu;

 MWApproxBatchInputs inputs{};
 inputs.area_cm2 = area_cm2.data();
 inputs.depth_cm = depth_cm.data();
 inputs.bmag_g = bmag_g.data();
 inputs.temperature_k = temperature_k.data();
 inputs.thermal_density_cm3 = thermal_density_cm3.data();
 inputs.nonthermal_density_cm3 = nonthermal_density_cm3.data();
 inputs.delta = delta.data();
 inputs.theta_deg = theta_deg.data();
 inputs.emin_mev = emin_mev.data();
 inputs.emax_mev = emax_mev.data();

 MWApproxBatchRlOutputs outputs{};
 outputs.status = status.data();
 outputs.freq_hz = freq_out;
 outputs.rl = rl_out;

 int res = MWApproxBatchRunRL(&config, &inputs, &outputs, d_sun_au);
 if (res != MW_BATCH_OK)
 {
  std::ostringstream message;
  message << "MWApproxBatchRunRL failed with status " << res;
  return ffi::Error::Internal(message.str());
 }

 for (int i = 0; i < batch_size; ++i)
 {
  if (status[i] != MW_BATCH_OK)
  {
   const size_t item_offset = static_cast<size_t>(i) * static_cast<size_t>(nfreq) *
                              static_cast<size_t>(MW_APPROX_RL_ROWS);
   std::fill(rl_out + item_offset, rl_out + item_offset + static_cast<size_t>(nfreq) * MW_APPROX_RL_ROWS, 0.0);
  }
 }

 return ffi::Error::Success();
}

}  // namespace mw_xla
