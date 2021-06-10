# NIH Chest X-Ray With PyTorch

## Requirements
* torch
* torchvision
* numpy

To run locally:
```
python main.py --data-path=/root/dir/to/dataset --model-path=/root/dir/to/save/models
```

Run: `python main.py --help` to get further options.  

To run the script on kaggle, create a notebook with the NIH Chest XRay dataset attached to it and add the following instructions:

```
!git clone https://github.com/PaulStryck/nih-chest-x-ray.git ./nih_chest_x_ray
!git -C nih_chest_x_ray pull
!pip install torchinfo

!python nih_chest_x_ray/main.py --data-path="/kaggle/input/data" --model-path="/kaggle/working/models" --test-bs=64 --train-bs=64 --val-bs=64 --data-frac=1 --log-interval=2 --save-interval=100
```

Adjust the --data-frac option between 0 and 1 to run only on a fraction of the data.
