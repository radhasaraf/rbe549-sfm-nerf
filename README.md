# Structure From Motion (SfM) and NeRF
Reconstruction of a 3d scene from a set of images with different view points (camera in motion)

## Results

### NeRF
<p float="middle">
<img src="phase2/outputs/lego_gif.gif" width="500" height="500"/>
</p>

<!-- #### Training Progress
<p float="middle">
<img src="phase2/outputs/loss_every_view.png" width="700" height="350"/>
</p>


<p float="middle">
<img src="phase2/outputs/loss_every_iter.png" width="700" height="350"/>
</p> -->

### Structure from Motion (SfM)

#### Matches using RANSAC

##### Before 
<p float="middle">
<img src="phase1/results/before_RANSAC_1_2.png"/>
</p>

##### After
<p float="middle">
<img src="phase1/results/after_RANSAC_1_2.png"/>
</p>


#### Epipolars
<p float="middle">
<img src="phase1/results/epipolars_1_2.png"/>
</p>



#### Camera Disambiguation using Chierality condition

##### Initial Triangulation
<p float="middle">
<img src="phase1/results/camera_disambiguation.png"/>
</p>

##### After Disambiguation
<p float="middle">
<img src="phase1/results/linear_triangulation.png"/>
</p>


#### NonLinear Triangulation
<p float="middle">
<img src="phase1/results/nonlinear_triangulation.png"/>
</p>

#### Camera Registration using Perspective-n-Points (PnP)

#### NonLinear PnP

#### Bundle Adjustment

## Collaborators
Sai Ramana Kiran - spinnamaraju@wpi.edu

Radha Saraf - rrsaraf@wpi.edu
