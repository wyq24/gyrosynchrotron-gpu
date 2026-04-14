#include <stdint.h>

#include <limits>

#include "XLAInterfaceCommon.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/api/api.h"

namespace ffi = xla::ffi;

namespace
{

ffi::Error MWApproxBatchRlCpuImpl(int32_t npoints,
                                  int32_t q_on,
                                  double nu0_hz,
                                  double dlog10_nu,
                                  double d_sun_au,
                                  ffi::Buffer<ffi::DataType::F64> params,
                                  ffi::ResultBuffer<ffi::DataType::F64> rl,
                                  ffi::ResultBufferR1<ffi::DataType::F64> freq)
{
 int64_t flat_batch_size_64 = 0;
 int64_t nfreq_64 = 0;
 if (ffi::Error err = mw_xla::ValidateBuffers(params, rl, freq, &flat_batch_size_64, &nfreq_64); err.failure())
  return err;

 if (flat_batch_size_64 > std::numeric_limits<int>::max())
  return ffi::Error::InvalidArgument("batch size exceeds int range");
 if (nfreq_64 > std::numeric_limits<int>::max())
  return ffi::Error::InvalidArgument("nfreq exceeds int range");

 return mw_xla::RunRlFromPackedParamsHost(MW_BATCH_BACKEND_CPU,
                                          npoints,
                                          q_on,
                                          nu0_hz,
                                          dlog10_nu,
                                          d_sun_au,
                                          static_cast<int>(flat_batch_size_64),
                                          static_cast<int>(nfreq_64),
                                          params.typed_data(),
                                          rl->typed_data(),
                                          freq->typed_data());
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MWApproxBatchRlCpu,
    MWApproxBatchRlCpuImpl,
    ffi::Ffi::Bind()
        .Attr<int32_t>("npoints")
        .Attr<int32_t>("q_on")
        .Attr<double>("nu0_hz")
        .Attr<double>("dlog10_nu")
        .Attr<double>("d_sun_au")
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::BufferR1<ffi::DataType::F64>>());
