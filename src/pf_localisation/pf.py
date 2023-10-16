from geometry_msgs.msg import Pose, PoseArray, Quaternion
from .pf_base import PFLocaliserBase
import math
import rospy
from .util import rotateQuaternion, getHeading
from random import random , gauss, uniform
from time import time

# To install scikit, write: pip3 install -U scikit-learn
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import distance

class PFLocaliser(PFLocaliserBase):
       
    def __init__(self):
        super(PFLocaliser, self).__init__()
        self.NUMBER_PREDICTED_READINGS = 20
        self.ODOM_ROTATION_NOISE = 0
        self.ODOM_TRANSLATION_NOISE = 0  
        self.ODOM_DRIFT_NOISE = 0

    def initialise_particle_cloud(self, initialpose):
        num_particles = 500  # Number of particles in the particle cloud

        particle_cloud = PoseArray()
        particle_cloud.header.frame_id = "map"  # Adjust the frame_id as needed

        for _ in range(num_particles):
            particle = Pose()

            particle.position.x = initialpose.pose.pose.position.x + gauss(0, 0.35)  # Adjust noise parameters as needed
            particle.position.y = initialpose.pose.pose.position.y + gauss(0, 0.35)  # Adjust noise parameters as needed

            initial_yaw = getHeading(initialpose.pose.pose.orientation)
            particle.orientation = rotateQuaternion(initialpose.pose.pose.orientation, gauss(0, 0.35))  # Adjust noise parameters as needed

            particle_cloud.poses.append(particle)

        return particle_cloud


    def update_particle_cloud(self, scan):
        weights = []

        
        for particle in self.particlecloud.poses:
            weight = self.sensor_model.get_weight(scan, particle)
            print("Input Laser Scan Data:", scan)
            print("Particle Pose:", particle)
            weights.append(weight)

        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        print("Total Weight:", total_weight)
        print("Normalized Weights:", normalized_weights)

        new_particle_cloud = PoseArray()
        new_particle_cloud.header.frame_id = "map"

        num_particles = len(self.particlecloud.poses)

        for _ in range(num_particles):
            rand_value = random()
            cumulative_weight = 0
            selected_particle = None

            for i in range(num_particles):
                cumulative_weight += normalized_weights[i]
                if cumulative_weight >= rand_value:
                    selected_particle = self.particlecloud.poses[i]
                    break

            new_particle = Pose()
            new_particle.position.x = selected_particle.position.x + gauss(0, 0.25)
            new_particle.position.y = selected_particle.position.y + gauss(0, 0.25)
            new_particle.orientation = rotateQuaternion(selected_particle.orientation, gauss(0, 0.25))

            new_particle_cloud.poses.append(new_particle)

        self.particlecloud = new_particle_cloud

    def estimate_pose(self):
        # return Pose()
        # rospy.loginfo("estimating")
        poses = self.particlecloud.poses
        coordinates = np.array([[p.position.x, p.position.y] for p in poses]) # Extract the x,y coordinates

        core_distance = .15 # Epsilon distance
        core_samples = 50 # Number of points needed to be considered "core"
        # ABOVE VALUES BOTH NEED TUNING

        db = DBSCAN(eps=core_distance, min_samples=core_samples).fit(coordinates) # Performs the clustering
        labels = db.labels_
        core_points_indices = db.core_sample_indices_

        # If there are no clusters identified, then we average all particles
        if not(labels.__contains__(0)):
            # rospy.loginfo("no clusters found, averaging all")
            estimated_point = Pose()
            for p in poses: 
                estimated_point.position.x += (p.position.x / len(poses))
                estimated_point.position.y += (p.position.y / len(poses))

                # Just average rotation
                estimated_point.orientation.x = (p.orientation.x / len(poses))
                estimated_point.orientation.y = (p.orientation.y / len(poses))
                estimated_point.orientation.z = (p.orientation.z / len(poses))
                estimated_point.orientation.w = (p.orientation.w / len(poses))
            return estimated_point
    
        
        # Remove all points except core points
        best_cluster = []
        for i in core_points_indices:
            # The densest cluster will be the one with the label of 0
            if labels[i] == 0:
                best_cluster.append(poses[i])

        x_coords = np.array([p.position.x for p in best_cluster])
        y_coords = np.array([p.position.y for p in best_cluster])

        estimated_point = Pose()
        estimated_point.position.x = np.mean(x_coords)
        estimated_point.position.y = np.mean(y_coords)

        # Perform DBScan on orientation
        # array of form [[x1, ...], [y1, ...], [z1, ...], [w1, ...]]
        orientations = np.array([[p.orientation.x for p in best_cluster], [p.orientation.y for p in best_cluster], [p.orientation.z for p in best_cluster], [p.orientation.w for p in best_cluster]])
        o_std = [np.std(o) for o in orientations] # standard deviations for 4 vals
        o_mean = [np.mean(o) for o in orientations] # mean for 4 vals
        ranges = [[o_mean[i] - 2 * o_std[i], o_mean[i] + 2 * o_std[i]] for i in range(0,4)] # get min and maxes that values should be in, in each of the dimensions (x, y, z, w)

        estimated_orientation = Quaternion()
        count = 0
        # Remove values that are more than 2 std away from mean
        for i in range(0, len(orientations[0])):
            if all([(ranges[j][0] <= orientations[j][i] <= ranges[j][1]) for j in range(0, 4)]): # only add val to cal if it is in 2 standard deviations of the mean
                estimated_orientation.x += orientations[0][i]
                estimated_orientation.y += orientations[1][i]
                estimated_orientation.z += orientations[2][i]
                estimated_orientation.w += orientations[3][i]
                count += 1

        estimated_orientation.x = estimated_orientation.x / count    #calculate averges
        estimated_orientation.y = estimated_orientation.y / count
        estimated_orientation.z = estimated_orientation.z / count
        estimated_orientation.w = estimated_orientation.w / count

        estimated_point.orientation = estimated_orientation

        return estimated_point
	    
    def re_intialisation(self):
	    num_particles = 500  # Number of particles in the particle cloud
	    for _ in range(num_particles):
		    particle.position.x = 6*(gauss(0, 1))  # Starts at 0,0 so not effected by start pos of robot
		    particle.position.y = 6*(gauss(0, 1))  # 6 times the variance is enough to cover the whole map
	    quat_tf = [0, 1, 0, 0]
            quat_msg = Quaternion(quat_tf[0], quat_tf[1], quat_tf[2], quat_tf[3])

            particle.orientation = rotateQuaternion(quat_msg, 6*(gauss(0, 1))) #Ignore start angle of robot 

	    particle_cloud.poses.append(particle)
	    
	    self.particle_cloud = new_particle_cloud

