# torchys
Tinkering with Pytorch to learn how it can speed things up

## Lessons Learned:

 - Using the add funciton with a scale value alpha is 2x as fast as separate operations on the CPU.
 ```
util = torch.add(torch.tensor([const]), var, alpha=coefficient) #faster on CPU
util = var * coeff + K # 2x slower on CPU!

 - Using the simpler approach of multiplying and adding separately is faster on the GPU for both intel and Mac.
   * When using the GPU, running the same operation more times adds significant speed improvements
   * Mac's GPU seems to win when running 10 at a time.
   * Add with alpha is significantly slower on mac GPU than CUDA

 - Optimizing and using add with alpha strategically is fast. I assume the earlier implementation was slower because it was doing more memory allocation on GPU.