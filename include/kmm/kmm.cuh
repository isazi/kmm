
#pragma once

class Stream {
  public:
    Stream();
    Stream(CUDA& device);
    ~Stream();
    // Return a CUDA stream
    cudaStream_t getStream(CUDA& device);

  private:
    cudaStream_t cuda_stream;
};
