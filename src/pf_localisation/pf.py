from geometry_msgs.msg import Pose, PoseArray, Quaternion
from .pf_base import PFLocaliserBase
import math
import rospy
from .util import rotateQuaternion, getHeading
from random import random , gauss, uniform
from time import time

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
            particle.orientation = rotateQuaternion(initialpose.pose.pose.orientation, gauss(0, 0.2))  # Adjust noise parameters as needed

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
            new_particle.position.x = selected_particle.position.x + gauss(0, 0.15)
            new_particle.position.y = selected_particle.position.y + gauss(0, 0.15)
            new_particle.orientation = rotateQuaternion(selected_particle.orientation, gauss(0, 0.15))

            new_particle_cloud.poses.append(new_particle)

        self.particlecloud = new_particle_cloud

    def estimate_pose(self):
        num_particles = len(self.particlecloud.poses)

        if num_particles == 0:
            estimated_pose = self.estimatedpose.pose
        else:
            avg_x = sum(p.position.x for p in self.particlecloud.poses) / num_particles
            avg_y = sum(p.position.y for p in self.particlecloud.poses) / num_particles

            avg_quaternions = [p.orientation for p in self.particlecloud.poses]
            avg_orientation = Quaternion()
            avg_orientation.x = sum(q.x for q in avg_quaternions) / num_particles
            avg_orientation.y = sum(q.y for q in avg_quaternions) / num_particles
            avg_orientation.z = sum(q.z for q in avg_quaternions) / num_particles
            avg_orientation.w = sum(q.w for q in avg_quaternions) / num_particles

            estimated_pose = Pose()
            estimated_pose.position.x = avg_x
            estimated_pose.position.y = avg_y
            estimated_pose.orientation = avg_orientation

        return estimated_pose
