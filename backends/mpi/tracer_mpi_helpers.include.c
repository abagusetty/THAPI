static pthread_once_t _init = PTHREAD_ONCE_INIT;
static __thread volatile int in_init = 0;
static volatile unsigned int _initialized = 0;

/* ---------------------------------------------------------------------------
 * GPU pointer classification
 *
 * Probes already-loaded GPU backend libraries through dlsym(RTLD_NOLOAD).
 * No GPU SDK headers are included here; all symbols are resolved as opaque
 * function pointers so the MPI tracer compiles unconditionally regardless of
 * which GPU backends are present on the system.
 *
 * Three backends are probed independently at _load_tracer() time:
 *   Level Zero  – zeMemGetAllocProperties (L0 1.0+, 2020)
 *   CUDA Driver – cuPointerGetAttribute   (CUDA driver 7.0+, 2015)
 *   HIP runtime – hipPointerGetAttributes (ROCm 1.6+, 2018)
 *
 * All three calls perform an O(1) driver-side allocation-table lookup;
 * the overhead is negligible compared to the MPI call itself.
 * ---------------------------------------------------------------------------
 */

typedef enum {
  MPI_PTR_HOST         = 0, /* ordinary malloc / stack / MPI_IN_PLACE / NULL */
  MPI_PTR_L0_DEVICE    = 1, /* zeMemAllocDevice */
  MPI_PTR_L0_SHARED    = 2, /* zeMemAllocShared (accessible from both sides) */
  MPI_PTR_L0_HOST      = 3, /* zeMemAllocHost   (pinned, GPU-accessible) */
  MPI_PTR_CUDA_DEVICE  = 4, /* cudaMalloc */
  MPI_PTR_CUDA_HOST    = 5, /* cudaMallocHost / pinned */
  MPI_PTR_CUDA_MANAGED = 6, /* cudaMallocManaged */
  MPI_PTR_HIP_DEVICE   = 7, /* hipMalloc */
  MPI_PTR_HIP_HOST     = 8, /* hipMallocHost / pinned */
  MPI_PTR_UNKNOWN      = 9, /* query succeeded but type unrecognised */
} mpi_ptr_type_t;

/*
 * Opaque function-pointer types — no GPU SDK headers required.
 *
 * Level Zero: zeMemGetAllocProperties
 *   ze_result_t (ze_context_handle_t, const void *,
 *                ze_memory_allocation_properties_t *, ze_device_handle_t *)
 *
 *   ze_memory_allocation_properties_t ABI (L0 1.0):
 *     uint32_t  stype   (+0)
 *     void     *pNext   (+8)
 *     uint32_t  type    (+16)   <-- ZE_MEMORY_TYPE_* we read
 *     uint64_t  id      (+24)
 *     uint64_t  pageSize(+32)
 *
 *   ZE_MEMORY_TYPE values: UNKNOWN=0, HOST=1, DEVICE=2, SHARED=3
 *   ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES = 0x1c
 *
 * CUDA: cuPointerGetAttribute
 *   CUresult (void *data, CUpointer_attribute attr, CUdeviceptr ptr)
 *   CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
 *   CUmemorytype: HOST=1, DEVICE=2, UNIFIED=4
 *
 * HIP: hipPointerGetAttributes
 *   hipError_t (hipPointerAttribute_t *attr, const void *ptr)
 *   hipPointerAttribute_t ABI:
 *     int  memoryType  (+0)   hipMemoryTypeHost=1, hipMemoryTypeDevice=2
 *     int  device      (+4)
 *     void *devicePointer (+8)
 *     void *hostPointer  (+16)
 *     int  isManaged   (+24)
 */
typedef int (*_thapi_ze_mem_props_fn)(void *, const void *, void *, void **);
typedef int (*_thapi_cu_ptr_attr_fn)(void *, int, unsigned long long);
typedef int (*_thapi_hip_ptr_attrs_fn)(void *, const void *);

static _thapi_ze_mem_props_fn _ze_mem_get_props = NULL;
static void                  *_ze_context       = NULL;
static _thapi_cu_ptr_attr_fn  _cu_ptr_get_attr  = NULL;
static _thapi_hip_ptr_attrs_fn _hip_ptr_get_attrs = NULL;

static inline mpi_ptr_type_t thapi_classify_pointer(const void *ptr) {
  /* MPI_IN_PLACE is (void*)1 in MPICH; skip GPU queries for sentinels. */
  if (!ptr || ptr == MPI_IN_PLACE)
    return MPI_PTR_HOST;

  /* --- Level Zero -------------------------------------------------------- */
  if (_ze_mem_get_props && _ze_context) {
    /*
     * ze_memory_allocation_properties_t stack buffer.
     * stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES (0x1c)
     * pNext = NULL, rest zeroed.
     */
    unsigned char props[40];
    __builtin_memset(props, 0, sizeof(props));
    *(uint32_t *)(props + 0) = 0x1cu; /* stype */
    if (_ze_mem_get_props(_ze_context, ptr, props, NULL) == 0 /*ZE_RESULT_SUCCESS*/) {
      uint32_t type = *(uint32_t *)(props + 16);
      switch (type) {
        case 2: return MPI_PTR_L0_DEVICE; /* ZE_MEMORY_TYPE_DEVICE */
        case 3: return MPI_PTR_L0_SHARED; /* ZE_MEMORY_TYPE_SHARED */
        case 1: return MPI_PTR_L0_HOST;   /* ZE_MEMORY_TYPE_HOST   */
        case 0: break; /* ZE_MEMORY_TYPE_UNKNOWN — plain host, try next */
        default: break;
      }
    }
  }

  /* --- CUDA Driver API --------------------------------------------------- */
  if (_cu_ptr_get_attr) {
    /*
     * cuPointerGetAttribute(&mem_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE=2, ptr)
     * Returns non-zero (CUDA_ERROR_INVALID_VALUE) for plain host pointers.
     */
    unsigned int mem_type = 0;
    if (_cu_ptr_get_attr(&mem_type, 2 /*CU_POINTER_ATTRIBUTE_MEMORY_TYPE*/,
                         (unsigned long long)(uintptr_t)ptr) == 0 /*CUDA_SUCCESS*/) {
      switch (mem_type) {
        case 2: return MPI_PTR_CUDA_DEVICE;  /* CU_MEMORYTYPE_DEVICE  */
        case 1: return MPI_PTR_CUDA_HOST;    /* CU_MEMORYTYPE_HOST    */
        case 4: return MPI_PTR_CUDA_MANAGED; /* CU_MEMORYTYPE_UNIFIED */
        default: return MPI_PTR_UNKNOWN;
      }
    }
    /* non-zero => plain host pointer, fall through */
  }

  /* --- HIP runtime ------------------------------------------------------- */
  if (_hip_ptr_get_attrs) {
    /*
     * hipPointerGetAttributes(&attrs, ptr)
     * attrs.memoryType at offset 0: 1=host, 2=device
     * Returns non-zero (hipErrorInvalidValue) for non-HIP pointers.
     */
    unsigned char attrs[32];
    __builtin_memset(attrs, 0, sizeof(attrs));
    if (_hip_ptr_get_attrs(attrs, ptr) == 0 /*hipSuccess*/) {
      int mem_type = *(int *)(attrs + 0);
      switch (mem_type) {
        case 2: return MPI_PTR_HIP_DEVICE; /* hipMemoryTypeDevice */
        case 1: return MPI_PTR_HIP_HOST;   /* hipMemoryTypeHost   */
        default: return MPI_PTR_UNKNOWN;
      }
    }
  }

  return MPI_PTR_HOST;
}

/*
 * Returns non-zero when the pointer targets GPU device memory (device or
 * shared/managed).  Pinned host allocations (L0_HOST, CUDA_HOST, HIP_HOST)
 * are intentionally excluded: they are CPU-accessible and MPI stacks may
 * handle them without GPU-aware paths.
 */
static inline int thapi_is_gpu_aware_ptr(const void *ptr) {
  mpi_ptr_type_t t = thapi_classify_pointer(ptr);
  return (t == MPI_PTR_L0_DEVICE   || t == MPI_PTR_L0_SHARED  ||
          t == MPI_PTR_CUDA_DEVICE  || t == MPI_PTR_CUDA_MANAGED ||
          t == MPI_PTR_HIP_DEVICE);
}

/* ---------------------------------------------------------------------------
 * Tracer initialisation
 * ---------------------------------------------------------------------------
 */

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
    if (ptr == (void *)&MPI_Init) { /* opening oneself */
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

  /* Resolve GPU pointer-query symbols from already-loaded backend libs.
   * RTLD_NOLOAD ensures we never pull in a GPU runtime that the application
   * did not already load; if a library is absent the pointer stays NULL and
   * all queries for that backend are silently skipped. */
  {
    void *h;

    /* Level Zero */
    h = dlopen("libze_loader.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (h) {
      _ze_mem_get_props =
          (_thapi_ze_mem_props_fn)(uintptr_t)dlsym(h, "zeMemGetAllocProperties");
      /* Retrieve the L0 context from the THAPI L0 tracer if co-loaded. */
      void *l0h = dlopen("libthapi_ze.so", RTLD_LAZY | RTLD_NOLOAD);
      if (l0h) {
        void *(*get_ctx)(void) = (void *(*)(void))(uintptr_t)dlsym(l0h, "_thapi_ze_context_get");
        if (get_ctx)
          _ze_context = get_ctx();
        dlclose(l0h);
      }
      dlclose(h);
    }

    /* CUDA Driver */
    h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (h) {
      _cu_ptr_get_attr =
          (_thapi_cu_ptr_attr_fn)(uintptr_t)dlsym(h, "cuPointerGetAttribute");
      dlclose(h);
    }

    /* HIP */
    h = dlopen("libamdhip64.so", RTLD_LAZY | RTLD_NOLOAD);
    if (h) {
      _hip_ptr_get_attrs =
          (_thapi_hip_ptr_attrs_fn)(uintptr_t)dlsym(h, "hipPointerGetAttributes");
      dlclose(h);
    }
  }
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
