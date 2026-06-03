static pthread_once_t _init = PTHREAD_ONCE_INIT;
static __thread volatile int in_init = 0;
static volatile unsigned int _initialized = 0;

/* ---------------------------------------------------------------------------
 * GPU pointer classification
 * Supported backends (each guarded by its own feature macro):
 *   THAPI_HAS_L0   – Level Zero  (ze_context_handle_t populated by L0 tracer)
 *   THAPI_HAS_CUDA – CUDA Driver API  (requires libcuda.so, CUDA 7.0+)
 *   THAPI_HAS_HIP  – HIP runtime API  (ROCm 3.5+, hipPointerGetAttributes)
 *
 * All three APIs perform an O(1) driver-side allocation-table lookup; the
 * cost is negligible compared to the MPI call itself.
 * ---------------------------------------------------------------------------
 */

#if defined(THAPI_HAS_L0)
#include <level_zero/ze_api.h>
/* ze_context populated by the L0 backend tracer at zeContextCreate time.
 * Declared extern so the L0 interception layer can assign it.
 * Only touched inside _init_tracer / after _initialized is set. */
extern ze_context_handle_t _thapi_ze_context;
#endif

#if defined(THAPI_HAS_CUDA)
#include <cuda.h>
#endif

#if defined(THAPI_HAS_HIP)
#include <hip/hip_runtime_api.h>
#endif

typedef enum {
  MPI_PTR_HOST        = 0,  /* ordinary malloc / stack / MPI_IN_PLACE / NULL */
  MPI_PTR_L0_DEVICE   = 1,  /* zeMemAllocDevice */
  MPI_PTR_L0_SHARED   = 2,  /* zeMemAllocShared (accessible from both sides) */
  MPI_PTR_L0_HOST     = 3,  /* zeMemAllocHost  (pinned, GPU-accessible) */
  MPI_PTR_CUDA_DEVICE = 4,  /* cudaMalloc */
  MPI_PTR_CUDA_HOST   = 5,  /* cudaMallocHost / pinned */
  MPI_PTR_CUDA_MANAGED= 6,  /* cudaMallocManaged */
  MPI_PTR_HIP_DEVICE  = 7,  /* hipMalloc */
  MPI_PTR_HIP_HOST    = 8,  /* hipMallocHost / pinned */
  MPI_PTR_UNKNOWN     = 9,  /* query succeeded but type unrecognised */
} mpi_ptr_type_t;

/* Classify a buffer pointer at MPI entry time.
 * Returns MPI_PTR_HOST for NULL and MPI_IN_PLACE (special MPI sentinels). */
static inline mpi_ptr_type_t thapi_classify_pointer(const void *ptr) {
  /* MPI_IN_PLACE is typically defined as (void*)1 in MPICH. */
  if (!ptr || ptr == MPI_IN_PLACE)
    return MPI_PTR_HOST;

#if defined(THAPI_HAS_L0)
  if (_thapi_ze_context) {
    ze_memory_allocation_properties_t props = {};
    props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    /* zeMemGetAllocProperties is available since Level Zero 1.0 (2020-06).
     * It returns ZE_RESULT_SUCCESS even for host pointers (type = UNKNOWN). */
    if (zeMemGetAllocProperties(_thapi_ze_context, ptr, &props, NULL)
        == ZE_RESULT_SUCCESS) {
      switch (props.type) {
        case ZE_MEMORY_TYPE_DEVICE: return MPI_PTR_L0_DEVICE;
        case ZE_MEMORY_TYPE_SHARED: return MPI_PTR_L0_SHARED;
        case ZE_MEMORY_TYPE_HOST:   return MPI_PTR_L0_HOST;
        case ZE_MEMORY_TYPE_UNKNOWN: break; /* fall through to next backend */
        default: break;
      }
    }
  }
#endif

#if defined(THAPI_HAS_CUDA)
  {
    /* CU_POINTER_ATTRIBUTE_MEMORY_TYPE available since CUDA 7.0 (Driver 346).
     * Returns CUDA_ERROR_INVALID_VALUE for non-CUDA (host) pointers, which
     * is the standard way to detect them – no separate "is CUDA ptr" API. */
    unsigned int mem_type = 0;
    CUresult res = cuPointerGetAttribute(
        &mem_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)ptr);
    if (res == CUDA_SUCCESS) {
      switch ((CUmemorytype)mem_type) {
        case CU_MEMORYTYPE_DEVICE:  return MPI_PTR_CUDA_DEVICE;
        case CU_MEMORYTYPE_HOST:    return MPI_PTR_CUDA_HOST;
        case CU_MEMORYTYPE_UNIFIED: return MPI_PTR_CUDA_MANAGED;
        default: return MPI_PTR_UNKNOWN;
      }
    }
    /* CUDA_ERROR_INVALID_VALUE => plain host pointer, continue */
  }
#endif

#if defined(THAPI_HAS_HIP)
  {
    /* hipPointerGetAttributes available since ROCm 1.6 / HIP 1.6.
     * Returns hipErrorInvalidValue for non-HIP pointers. */
    hipPointerAttribute_t attrs;
    if (hipPointerGetAttributes(&attrs, ptr) == hipSuccess) {
      switch (attrs.memoryType) {
        case hipMemoryTypeDevice: return MPI_PTR_HIP_DEVICE;
        case hipMemoryTypeHost:   return MPI_PTR_HIP_HOST;
        default: return MPI_PTR_UNKNOWN;
      }
    }
  }
#endif

  return MPI_PTR_HOST;
}

/* Convenience: returns non-zero when the pointer lives on a GPU device
 * (device or shared/managed, but NOT pinned host). Use this flag for the
 * gpu_aware boolean field in the trace interval model. */
static inline int thapi_is_gpu_aware_ptr(const void *ptr) {
  mpi_ptr_type_t t = thapi_classify_pointer(ptr);
  return (t == MPI_PTR_L0_DEVICE  || t == MPI_PTR_L0_SHARED ||
          t == MPI_PTR_CUDA_DEVICE || t == MPI_PTR_CUDA_MANAGED ||
          t == MPI_PTR_HIP_DEVICE);
}

/* ---------------------------------------------------------------------------
 * Tracer initialisation (unchanged logic, extended for GPU context)
 * ---------------------------------------------------------------------------
 */

#if defined(THAPI_HAS_L0)
ze_context_handle_t _thapi_ze_context = NULL;
#endif

static void _load_tracer(void) {
  char *s = NULL;
  void *handle = NULL;
  int verbose = 0;

  s = getenv("LTTNG_UST_MPI_LIBMPI");
  if (s)
    handle = dlopen(s, RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND);
  else
    handle = dlopen("libmpi.so", RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND);
  if (handle) {
    void *ptr = dlsym(handle, "MPI_Init");
    if (ptr == (void *)&MPI_Init) { // opening oneself
      dlclose(handle);
      handle = NULL;
    }
  }

  if (!handle) {
    fprintf(stderr, "THAPI: Failure: could not load MPI library!\n");
    exit(1);
  }

  s = getenv("LTTNG_UST_MPI_VERBOSE");
  if (s)
    verbose = 1;

  find_mpi_symbols(handle, verbose);

#if defined(THAPI_HAS_L0)
  /* Attempt to retrieve a Level Zero context from the L0 tracer's shared
   * state. The symbol _thapi_ze_context_get() is optionally exported by the
   * L0 backend shared library when both backends are loaded together. */
  {
    void *l0_handle = dlopen("libthapi_ze.so", RTLD_LAZY | RTLD_NOLOAD);
    if (l0_handle) {
      ze_context_handle_t (*get_ctx)(void) =
          (ze_context_handle_t (*)(void))dlsym(l0_handle, "_thapi_ze_context_get");
      if (get_ctx)
        _thapi_ze_context = get_ctx();
      dlclose(l0_handle);
    }
  }
#endif
}

static inline void _init_tracer(void) {
  if (__builtin_expect(_initialized, 1))
    return;
  /* Avoid reentrancy */
  if (!in_init) {
    in_init = 1;
    __sync_synchronize();
    pthread_once(&_init, _load_tracer);
    __sync_synchronize();
    in_init = 0;
  }
  _initialized = 1;
}
