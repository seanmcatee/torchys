# torchys
Tinkering with Pytorch to learn how it can speed things up

## Lessons Learned:

 - Using the add funciton with a scale value alpha is 2x as fast as separate operations on the CPU.
 ```
util = torch.add(torch.tensor([const]), var, alpha=coefficient) #faster on CPU
util = var * coeff + K # 2x slower on CPU!
