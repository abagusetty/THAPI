require_relative 'gen_mpi_library_base'

# Buffer-bearing parameter names that indicate a primary data buffer.
# When any of these appear in a function's parameter list, the generated
# wrapper will call thapi_is_gpu_aware_ptr() on the first one found and
# emit the result as a synthetic `gpu_aware` field in the LTTng tracepoint.
GPU_AWARE_BUF_PARAMS = %w[
  buf sendbuf recvbuf buffer origin_addr result_addr compare_addr
  inbuf outbuf base baseptr
].freeze

class MPILibrary < MPILibraryBase

  # Returns the name of the first buffer parameter for the given function,
  # or nil if the function does not operate on a data buffer.
  def gpu_aware_buf_param(function)
    function.parameters.find do |p|
      GPU_AWARE_BUF_PARAMS.include?(p.name.to_s)
    end&.name
  end

  def generate_wrapper(function)
    buf_param = gpu_aware_buf_param(function)
    super(function) do |f|
      if buf_param
        # Classify pointer at entry and store in thread-local for the
        # exit tracepoint to pick up via entries_gpu_aware_callback.
        <<~C
          /* GPU-aware classification: classify #{buf_param} */
          int8_t _gpu_aware = (int8_t)thapi_is_gpu_aware_ptr(
              (const void *)(uintptr_t)#{buf_param});
          tracepoint(lttng_ust_mpi, #{f.name}_entry_gpu_aware,
                     #{f.parameters.map(&:name).join(', ')}, _gpu_aware);
        C
      end
    end
  end

end
