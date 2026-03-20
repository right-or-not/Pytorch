

Tensor: 
- `(B, C, H, W)`
- `(batch, channels, height, width)`


Kernel: 
- `(C, H, W)`
- `(in_channels, kernel_height, kernel_width)`
- Kernel Size just change the Height and Width of Tensor


Convolutional Layer:
- `(C, C, H, W)`
- `(out_channels, in_channels, kernel_height, kernel_width)`
- `(batch, in_channels, H, W)` => `(batch, out_channels, H_out, W_out)`
- After Convolute Kernel: `(in_channels, H, W)` => `(1, H_out, W_out)`
- Combine All Concolution Results: `(1, H_out, W_out)` => `(out_channels, H_out, W_out)`

Relationship between `(H, W)` and `(H_out, W_out)`:
- `H_out = H + 2 * padding - kernel_height + 1`
- `W_out = W + 2 * padding - kernel_width + 1`
