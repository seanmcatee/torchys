# torchys
Tinkering with Pytorch to learn how it can speed things up

## Lessons Learned:

 - Using the add funciton with a scale value alpha is 2x faster than separate operations.
   * The addition can start with a simple scalar variable - it doesn't need to be converted to a tensor first.
   * Converting to a tensor in the add statement seems to add significant penalty on GPU.
 ```python
util = torch.add(const, var, alpha=coeff) #faster on CPU
util = var * coeff + K # 2x slower on CPU!
```

 - Each operation adds time!  Even a simple operation that converts a scalar to a one-element tensor adds a small amount of overhead, more noticable on the GPU.

 - When using the GPU, running the same operation more times adds significant speed improvements
   * Mac's GPU seems to win when running 10 at a time.

 TODO:
  - Test on other machines?
  - Test logsums and exponentiation (inc. logsum functions that do both)