/*
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef _WIN32
#include <unistd.h>
#endif

#include <thread>
#include "rpp/device_name.hpp"
#include "rpp/errors.hpp"
#include "rpp/logger.hpp"
#include "rpp/handle.hpp"
#include "rpp/kernel_cache.hpp"
#include "rpp/binary_cache.hpp"

namespace rpp {

// Get current context
// We leak resources for now as there is no hipCtxRetain API
hipCtx_t get_ctx()
{
    hipInit(0);
    hipCtx_t ctx;
    auto status = 0;
    if(status != hipSuccess)
        RPP_THROW("No device");
    return ctx;
}

std::size_t GetAvailableMemory()
{
    size_t free, total;
    auto status = hipMemGetInfo(&free, &total);
    if(status != hipSuccess)
        RPP_THROW_HIP_STATUS(status, "Failed getting available memory");
    return free;
}

void* default_allocator(void*, size_t sz)
{
    if(sz > GetAvailableMemory())
        RPP_THROW("Memory not available to allocate buffer: " + std::to_string(sz));
    void* result;
    auto status = hipMalloc(&result, sz);
    if(status != hipSuccess)
    {
        status = hipHostMalloc(&result, sz);
        if(status != hipSuccess)
            RPP_THROW_HIP_STATUS(status, "Hip error creating buffer " + std::to_string(sz) + ": ");
    }
    return result;
}

void default_deallocator(void*, void* mem)
{
    CHECK_RETURN_STATUS(hipFree(mem));
}

int get_device_id() // Get random device
{
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
        RPP_THROW("No device");
    return device;
}

void set_ctx(hipCtx_t ctx)
{
    auto status =  0;
    if(status != hipSuccess)
        RPP_THROW("Error setting context");
}

struct HandleImpl
{
    using StreamPtr = std::shared_ptr<typename std::remove_pointer<hipStream_t>::type>;

    hipCtx_t ctx;
    StreamPtr stream = nullptr;
    int device = -1;
    Allocator allocator{};
    KernelCache cache;
    bool enable_profiling = false;
    float profiling_result = 0.0;
    size_t nBatchSize = 1;
    Rpp32u numThreads = 0;
    InitHandle* initHandle = nullptr;

    HandleImpl() : ctx(get_ctx()) {}

    static StreamPtr reference_stream(hipStream_t s)
    {
        return StreamPtr{s, null_deleter{}};
    }

    void set_ctx()
    {
        rpp::set_ctx(this->ctx);
        // rpp::set_device(this->device);
        // Check device matches
        if(this->device != get_device_id())
            RPP_THROW("Running handle on wrong device");
    }

    void PreInitializeBufferCPU()
    {
        this->initHandle = new InitHandle();

        this->initHandle->nbatchSize = this->nBatchSize;

        for(int i = 0; i < 10; i++)
        {
            this->initHandle->mem.mcpu.floatArr[i].floatmem = (Rpp32f *)malloc(sizeof(Rpp32f) * this->nBatchSize);
            this->initHandle->mem.mcpu.uintArr[i].uintmem = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
            this->initHandle->mem.mcpu.intArr[i].intmem = (Rpp32s *)malloc(sizeof(Rpp32s) * this->nBatchSize);
            this->initHandle->mem.mcpu.ucharArr[i].ucharmem = (Rpp8u *)malloc(sizeof(Rpp8u) * this->nBatchSize);
            this->initHandle->mem.mcpu.charArr[i].charmem = (Rpp8s *)malloc(sizeof(Rpp8s) * this->nBatchSize);
        }

        this->initHandle->mem.mcpu.rgbArr.rgbmem = (RpptRGB *)malloc(sizeof(RpptRGB) * this->nBatchSize);
        this->initHandle->mem.mcpu.scratchBufferHost = (Rpp32f *)malloc(sizeof(Rpp32f) * 99532800 * this->nBatchSize); // 7680 * 4320 * 3
    }

    void PreInitializeBuffer()
    {
        this->initHandle = new InitHandle();
        this->PreInitializeBufferCPU();

        for(int i = 0; i < 10; i++)
        {
            CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.floatArr[i].floatmem), sizeof(Rpp32f) * this->nBatchSize));
            CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.uintArr[i].uintmem), sizeof(Rpp32u) * this->nBatchSize));
            CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.intArr[i].intmem), sizeof(Rpp32s) * this->nBatchSize));
            CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.ucharArr[i].ucharmem), sizeof(Rpp8u) * this->nBatchSize));
            CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.charArr[i].charmem), sizeof(Rpp8s) * this->nBatchSize));
            CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.float3Arr[i].floatmem), sizeof(Rpp32f) * this->nBatchSize * 3));
        }

        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.rgbArr.rgbmem), sizeof(RpptRGB) * this->nBatchSize));
#ifdef AUDIO_SUPPORT
        // If AUDIO_SUPPORT is enabled, 'scratchBufferHip' needed to run RNNT training successfully are larger.
        // Current max allocation size = sizeof(Rpp32f) * 372877312, which is based on Spectrogram requirements
        // 1. Spectrogram requirements:
        //      - 372877312 = (512 * 3754 * 192) + (512 * 3754 * 2)
        //      - Above is the maximum scratch memory required for Spectrogram HIP kernel used in RNNT training (uses a batchsize 192)
        //      - (512 * 3754 * 192) is the maximum size that will be required for window output based on Librispeech dataset in RNNT training
        //      - (512 * 3754 * 2) is the size required for storing sin and cos coefficients required for FFT computation in Spectrogram HIP kernel in RNNT training
        // 2. Non Silent Region Detection requirements:
        //      - 115293120 = (600000 + 293 + 192) * 192
        //      - Above is the maximum scratch memory required for Non Silent Region Detection HIP kernel used in RNNT training (uses a batchsize 192)
        //      - 600000 is the maximum size that will be required for MMS buffer based on Librispeech dataset
        //      - 293 is the size required for storing reduction outputs for 600000 size sample
        //      - 192 is the size required for storing cutOffDB values for batch size 192
        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.scratchBufferHip.floatmem), sizeof(Rpp32f) * 372877312));
#else
        CHECK_RETURN_STATUS(hipMalloc(&(this->initHandle->mem.mgpu.scratchBufferHip.floatmem), sizeof(Rpp32f) * 8294400));   // 3840 x 2160
#endif
        CHECK_RETURN_STATUS(hipHostMalloc(&(this->initHandle->mem.mgpu.scratchBufferPinned.floatmem), sizeof(Rpp32f) * 8294400));    // 3840 x 2160
    }
};

Handle::Handle(rppAcceleratorQueue_t stream, size_t batchSize) : impl(new HandleImpl())
{
    impl->nBatchSize = batchSize;
    this->impl->device = get_device_id();
    this->impl->ctx = get_ctx();

    if(stream == nullptr)
        this->impl->stream = HandleImpl::reference_stream(nullptr);
    else
        this->impl->stream = HandleImpl::reference_stream(stream);

    this->SetAllocator(nullptr, nullptr, nullptr);
    impl->PreInitializeBuffer();
    RPP_LOG_I(*this);
}

Handle::Handle(size_t batchSize, Rpp32u numThreads) : impl(new HandleImpl())
{
    impl->nBatchSize = batchSize;
    numThreads = std::min(numThreads, std::thread::hardware_concurrency());
    if(numThreads == 0)
        numThreads = batchSize;
    impl->numThreads = numThreads;
    this->SetAllocator(nullptr, nullptr, nullptr);
    impl->PreInitializeBufferCPU();
}

Handle::~Handle() {}

void Handle::rpp_destroy_object_gpu()
{
    this->rpp_destroy_object_host();

    for(int i = 0; i < 10; i++)
    {
        CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.floatArr[i].floatmem));
        CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.uintArr[i].uintmem));
        CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.intArr[i].intmem));
        CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.ucharArr[i].ucharmem));
        CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.charArr[i].charmem));
        CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.float3Arr[i].floatmem));
    }

    CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.rgbArr.rgbmem));
    CHECK_RETURN_STATUS(hipFree(this->GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem));
    CHECK_RETURN_STATUS(hipHostFree(this->GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem));
}

void Handle::rpp_destroy_object_host()
{

    for(int i = 0; i < 10; i++)
    {
        free(this->GetInitHandle()->mem.mcpu.floatArr[i].floatmem);
        free(this->GetInitHandle()->mem.mcpu.uintArr[i].uintmem);
        free(this->GetInitHandle()->mem.mcpu.intArr[i].intmem);
        free(this->GetInitHandle()->mem.mcpu.ucharArr[i].ucharmem);
        free(this->GetInitHandle()->mem.mcpu.charArr[i].charmem);
    }

    free(this->GetInitHandle()->mem.mcpu.rgbArr.rgbmem);
    free(this->GetInitHandle()->mem.mcpu.scratchBufferHost);
}

size_t Handle::GetBatchSize() const
{
    return this->impl->nBatchSize;
}

Rpp32u Handle::GetNumThreads() const
{
    return this->impl->numThreads;
}

void Handle::SetBatchSize(size_t bSize) const
{
    this->impl->nBatchSize = bSize;
}

rppAcceleratorQueue_t Handle::GetStream() const
{
    return impl->stream.get();
}

InitHandle* Handle::GetInitHandle() const
{
    return impl->initHandle;
}

void Handle::SetAllocator(rppAllocatorFunction allocator, rppDeallocatorFunction deallocator, void* allocatorContext) const
{
    this->impl->allocator.allocator = allocator == nullptr ? default_allocator : allocator;
    this->impl->allocator.deallocator = deallocator == nullptr ? default_deallocator : deallocator;
    this->impl->allocator.context = allocatorContext;
}

Program Handle::LoadProgram(const std::string& program_name,
                            std::string params,
                            bool is_kernel_str,
                            const std::string& kernel_src)
{
    this->impl->set_ctx();

    params += " -mcpu=" + this->GetDeviceName();
    auto cache_file =
        rpp::LoadBinary(this->GetDeviceName(), program_name, params, is_kernel_str);
    if(cache_file.empty())
    {
        auto p =
            HIPOCProgram{program_name, params, is_kernel_str, this->GetDeviceName(), kernel_src};

        return p;
    }
    else
    {
        return HIPOCProgram{program_name, cache_file};
    }
}

std::size_t Handle::GetLocalMemorySize()
{
    int result;
    auto status = hipDeviceGetAttribute(
        &result, hipDeviceAttributeMaxSharedMemoryPerBlock, this->impl->device);
    if(status != hipSuccess)
        RPP_THROW_HIP_STATUS(status);

    return result;
}


std::string Handle::GetDeviceName()
{
    hipDeviceProp_t props{};
    hipGetDeviceProperties(&props, this->impl->device);
    std::string name(props.gcnArchName);
    return name;
}

std::ostream& Handle::Print(std::ostream& os) const
{
    return os;
}


} // namespace rpp
