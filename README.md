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
   * Adding a single first run before timing the operation results in faster and more consistent time results. Ramp-up must be an issue.

 - For logsums, it's faster to just write out the calculations, but could be numerically unstable.
   * Our current implementation just runs calcs, and does exp() in an outer loop for efficiency.
   * Probably want to keep this approach.

 - There was a slight time reduction if the logsums (written out) are in a single tensor referenced by index, rather than two vars. Not sure if this is meaningful in a more general sense.

 TODO:
  - Test on other machines?
