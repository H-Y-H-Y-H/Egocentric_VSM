# Egocentric Visual Self-Modeling for Autonomous Robot Dynamics Prediction and Adaptation (Under Review)

This repository contains the supplementary materials for the paper "Egocentric Visual Self-Modeling for Autonomous Robot Dynamics Prediction and Adaptation".

## Supplementary Materials

- `EgocentricVSM_SI.pdf`: Detailed supplementary information.

## Installation

Ensure you have Python installed on your system, and then install the dependencies specified in `requirements.txt`.

```
pip install -r requirements.txt
```

## Trained Models
Access our pre-trained models, including the standard robot model, the damaged robot model, 
and the Visual Odometry (VO) model, through the following 
link: [Download Models](https://www.dropbox.com/scl/fo/t8i734kphgv3tl41f39qd/h?rlkey=grxtf5yqa17o75b1hhr8dlj08&dl=0
)

## Usage
Navigate to the project directory and run main.py with the following mode:

1: Collect Data in the simulation.

2: Train the Egocentric Visual Self-Model.

3: Train the Visual Odometry (VO) Model.

4: Test the Egocentric Visual Self-Model

5: Recovery Test

6: Use VO to collect data


## Citation
If you find our work useful in your research, please consider citing:

@article{hu2023self,
  title={Self-supervised robot self-modeling using a single egocentric camera},
  author={Hu, Yuhang and Chen, Boyuan and Lipson, Hod},
  year={2023}
}


