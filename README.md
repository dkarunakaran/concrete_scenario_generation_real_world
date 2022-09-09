# concrete_scenario_generation_real_world


## Detailed explanation of how OpenSCENARIO and OpenDRIVE files are created 
### Lane Construction

There are two separate stacks for the lane construction: one for clustering the lane points belonging to the same lane marking and another for grouping the lane points clusters that belong to the same lane. Firstly, on each lidar scan, we filter the road and non-road points, and then we employ an intensity-based approach to filter out the lane and non-lane points. In order to cluster the lane points belonging to the same lane marking, we need to convert the lane points from the base_link frame to the Odom frame. Once the newly scanned lidar points are in the Odom frame, we loop through them to determine which of the existing clusters in the first stack is closer. We have a threshold distance to determine a lane point that belongs to a cluster. If the distance between the new lane point and the nearest point from the closest cluster is less than one meter, the new lane point is added to this cluster. Otherwise, the new lane point will be added as a new cluster in the stack and increase other clusters' inactive count by 1. The inactive count is reset to zero when adding a new lane point.  

We move a particular cluster into the second stack if the cluster's inactive count reaches the threshold limit, as shown in  equation~\ref{inactive_counter}. It indicates that a particular cluster is not receiving any lane point for a while, suggesting that it has received all the lane points of a lane marking.
```math

\oint_{a}^{b}
 
```
