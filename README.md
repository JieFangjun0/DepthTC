# DepthTC
step 1.Prepare image depth data. Generate image sequences and ground truth using the script in prepate/etc/data_prepare.py.
step 2.Obtain inputs from single-image_based depth estimation model such as MIDAS or others.
step 3.Place the data in the following format in the data folder:
data
   --train
     --GT
       --62763_pbz98_3D_MPO_70pc
         --0.png
         --1.png
         --...
       --62871_z98_Delaunay-Belleville1914-17F
         --...
       --...
     --MIDAS
       --62763_pbz98_3D_MPO_70pc
       --62871_z98_Delaunay-Belleville1914-17F
       --...
     --video
       --...
   --test
step 4.Train using the script in train.py.
