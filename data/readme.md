## Data Directory Structure

The data directory should be organized as follows:

```text
data/
├── images/   # 2D CT slices (input images)
└── masks/    # corresponding segmentation masks (if available)
```
### Notes
All images should be stored as 2D .png files
Image and mask filenames should be consistent for correct pairing
Masks are optional for test data but required for evaluation
