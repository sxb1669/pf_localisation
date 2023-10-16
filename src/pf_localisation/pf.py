from geometry_msgs.msg import Pose, PoseArray, Quaternion
from .pf_base import PFLocaliserBase
import math
import rospy
from .util import rotateQuaternion, getHeading
from random import random , gauss, uniform
from time import time
# Update test 
class PFLocaliser(PFLocaliserBase):
       
    def __init__(self):
        super(PFLocaliser, self).__init__()
        self.NUMBER_PREDICTED_READINGS = 20

    def initialise_particle_cloud(self, initialpose):
    """
        Set particle cloud to initialpose plus noise

        Called whenever an initialpose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        self.particlecloud can be initialised here. Initial pose of the robot
        is also set here.
        
        :Args:
            | initialpose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """
        num_particles = 100
        particle_cloud = PoseArray()
        particle_cloud.header.frame_id = "map"
        for _ in range(num_particles):
            particle = Pose()
            particle.position.x = initialpose.pose.pose.position.x + gauss(0, 0.1)
            particle.position.y = initialpose.pose.pose.position.y + gauss(0, 0.1)
            initial_yaw = getHeading(initialpose.pose.pose.orientation)
            particle.orientation = rotateQuaternion(initialpose.pose.pose.orientation, uniform(-0.1, 0.1))
            particle_cloud.poses.append(particle)
        return particle_cloud

    def update_particle_cloud(self, scan):
    """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

         """
        with self._update_lock:
            particle_weights = []
            for particle in self.particlecloud.poses:
                weight = self.sensor_model.get_weight(scan, particle)
                particle_weights.append(weight)
            total_weight = sum(particle_weights)
            normalized_weights = [w / total_weight for w in particle_weights]
            num_particles = len(self.particlecloud.poses)
            indices = np.arange(num_particles)
            selected_indices = np.random.choice(indices, size=num_particles, p=normalized_weights)
            resampled_particles = [self.particlecloud.poses[i] for i in selected_indices]
            for i in range(num_particles):
                noise_x = np.random.normal(0, 0.1)
                noise_y = np.random.normal(0, 0.1)
                noise_yaw = np.random.normal(0, 0.1)
                resampled_particles[i].position.x += noise_x
                resampled_particles[i].position.y += noise_y
                resampled_particles[i].orientation = rotateQuaternion(resampled_particles[i].orientation, noise_yaw)
            self.particlecloud.poses = resampled_particles
            self.particlecloud.header.stamp = rospy.Time.now()

    def estimate_pose(self):
    """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud).
        
        Create new estimated pose, given particle cloud
        E.g. just average the location and orientation values of each of
        the particles and return this.
        
        Better approximations could be made by doing some simple clustering,
        e.g. taking the average location of half the particles after 
        throwing away any which are outliers

        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
         """
        if not self.particlecloud.poses:
            estimated_pose = Pose()
            estimated_pose.orientation = rotateQuaternion(Quaternion(w=1.0), self.INIT_HEADING)
            estimated_pose.position.x = self.INIT_X
            estimated_pose.position.y = self.INIT_Y
            return estimated_pose

        positions = np.array([(pose.position.x, pose.position.y) for pose in self.particlecloud.poses])
        orientations = np.array([getHeading(pose.orientation) for pose in self.particlecloud.poses])

        estimated_pose = Pose()
        estimated_pose.position.x = np.mean(positions[:, 0])
        estimated_pose.position.y = np.mean(positions[:, 1])
        estimated_orientation = np.mean(np.exp(1j * orientations))
        estimated_pose.orientation = rotateQuaternion(Quaternion(w=1.0), np.angle(estimated_orientation))

        return estimated_pose
