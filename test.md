# Image Quality Metrics: SSIM and L2 Norm

## Introduction
In image processing and computer vision, two widely used metrics for evaluating image quality and similarity are the Structural Similarity Index (SSIM) and L2 Norm (Euclidean Distance). These metrics serve different purposes and provide complementary information about image quality.

## Mathematical Formulation

### SSIM
The Structural Similarity Index (SSIM) measures the perceived quality between two images by considering luminance, contrast, and structure:

$SSIM(x,y) = [l(x,y)]^\alpha \cdot [c(x,y)]^\beta \cdot [s(x,y)]^\gamma$

where:
- $l(x,y) = \frac{2\mu_x\mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1}$ (luminance comparison)
- $c(x,y) = \frac{2\sigma_x\sigma_y + C_2}{\sigma_x^2 + \sigma_y^2 + C_2}$ (contrast comparison)
- $s(x,y) = \frac{\sigma_{xy} + C_3}{\sigma_x\sigma_y + C_3}$ (structure comparison)
- $\alpha, \beta, \gamma$ are weights typically set to 1
- $C_1, C_2, C_3$ are constants to avoid division by zero

### L2 Norm
The L2 Norm (Euclidean Distance) measures the pixel-wise difference between two images:

$L2(x,y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$

where:
- $x_i, y_i$ are pixel values at position $i$
- $n$ is the total number of pixels

## Comparative Analysis

Both metrics offer distinct advantages in image quality assessment:

1. **SSIM (range: [-1, 1])**
   - Considers structural information
   - Better correlates with human visual perception
   - More computationally intensive
   - Value of 1 indicates perfect structural similarity

2. **L2 Norm (range: [0, ∞))**
   - Simple and straightforward calculation
   - Measures absolute pixel-wise differences
   - Computationally efficient
   - Value of 0 indicates identical images

## Implementation
```python
def calculate_metrics(generated_image, target_image):
    """
    Calculate both SSIM and L2 norm for image comparison
    
    Args:
        generated_image: Generated image tensor [-1,1]
        target_image: Target image tensor [-1,1]
    Returns:
        ssim_score: SSIM score in range [-1,1]
        l2_norm: L2 norm (Euclidean distance)
    """
    # Convert from [-1,1] to [0,1] range
    generated_image = (generated_image + 1) * 0.5
    target_image = (target_image + 1) * 0.5
    
    # Convert to numpy arrays
    generated_np = tf.clip_by_value(generated_image, 0, 1).numpy()
    target_np = tf.clip_by_value(target_image, 0, 1).numpy()
    
    # Calculate SSIM
    ssim_score = ssim(target_np, generated_np, 
                      data_range=1.0,
                      multichannel=True)
    
    # Calculate L2 norm
    l2_norm = np.linalg.norm(target_np - generated_np)
    
    return ssim_score, l2_norm
```

## Applications
These metrics are commonly used in:
- Image quality assessment
- Image compression evaluation
- Image reconstruction validation
- GAN performance evaluation
- Image restoration assessment

In practice, using both metrics provides a more comprehensive evaluation of image quality, as SSIM captures perceptual similarity while L2 Norm measures pixel-level accuracy. For our current model evaluation (SSIM: 0.6349, L2 Norm: 56.3501), the results indicate moderate structural similarity with room for improvement in pixel-level accuracy.
