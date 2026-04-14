#include <stdint.h>

#include <limits>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>

#include "XLAInterfaceCommon.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/api/api.h"

namespace ffi = xla::ffi;

namespace
{

ffi::Error CudaStatusToError(const char *op, cudaError_t status)
{
 std::ostringstream message;
 message << op << " failed: " << cudaGetErrorString(status);
 return ffi::Error::Internal(message.str());
}

ffi::Error MWApproxBatchRlCudaImpl(int32_t npoints,
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

 const size_t flat_batch_size = static_cast<size_t>(flat_batch_size_64);
 const size_t nfreq = static_cast<size_t>(nfreq_64);
 const size_t param_elements = flat_batch_size * static_cast<size_t>(mw_xla::MW_APPROX_PARAM_DIM);
 const size_t rl_elements = flat_batch_size * nfreq * static_cast<size_t>(mw_xla::MW_APPROX_RL_ROWS);

 std::vector<double> host_params(param_elements);
 std::vector<double> host_rl(rl_elements);
 std::vector<double> host_freq(nfreq);

 // The first CUDA FFI milestone is validation-first: stage device buffers
 // through the already-supported host-side MWApproxBatchRunRL CUDA path.
 cudaError_t status = cudaMemcpy(host_params.data(),
                                 params.typed_data(),
                                 sizeof(double) * param_elements,
                                 cudaMemcpyDeviceToHost);
 if (status != cudaSuccess) return CudaStatusToError("cudaMemcpy(params D2H)", status);

 if (ffi::Error err = mw_xla::RunRlFromPackedParamsHost(MW_BATCH_BACKEND_CUDA,
                                                        npoints,
                                                        q_on,
                                                        nu0_hz,
                                                        dlog10_nu,
                                                        d_sun_au,
                                                        static_cast<int>(flat_batch_size_64),
                                                        static_cast<int>(nfreq_64),
                                                        host_params.data(),
                                                        host_rl.data(),
                                                        host_freq.data());
     err.failure())
  return err;

 status = cudaMemcpy(freq->typed_data(),
                     host_freq.data(),
                     sizeof(double) * nfreq,
                     cudaMemcpyHostToDevice);
 if (status != cudaSuccess) return CudaStatusToError("cudaMemcpy(freq H2D)", status);

 status = cudaMemcpy(rl->typed_data(),
                     host_rl.data(),
                     sizeof(double) * rl_elements,
                     cudaMemcpyHostToDevice);
 if (status != cudaSuccess) return CudaStatusToError("cudaMemcpy(rl H2D)", status);

 return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MWApproxBatchRlCuda,
    MWApproxBatchRlCudaImpl,
    ffi::Ffi::Bind()
        .Attr<int32_t>("npoints")
        .Attr<int32_t>("q_on")
        .Attr<double>("nu0_hz")
        .Attr<double>("dlog10_nu")
        .Attr<double>("d_sun_au")
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::BufferR1<ffi::DataType::F64>>());
