#!/usr/bin/env python3
import rospy
import tf
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped
from tf.transformations import euler_from_quaternion
import numpy as np

class PathFollower:
    def __init__(self):
        rospy.init_node('path_follower')
        
        # 参数设置
        self.waypoint_reach_distance = rospy.get_param('~waypoint_reach_distance', 0.2)  # 米
        self.linear_velocity = rospy.get_param('~linear_velocity', 0.3)  # 米/秒
        self.angular_velocity = rospy.get_param('~angular_velocity', 0.5)  # 弧度/秒
        self.robot_frame = rospy.get_param('~robot_frame', 'base_link')
        
        # 目标路径
        self.path = None
        self.current_waypoint_index = 0
        
        # 发布器
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.current_target_pub = rospy.Publisher('/current_target', PoseStamped, queue_size=1)
        
        # 订阅器
        rospy.Subscriber('/planned_path', Path, self.path_callback)
        
        # TF监听器
        self.tf_listener = tf.TransformListener()
        
        # 控制循环
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_callback)
        
        rospy.loginfo("路径跟随器已初始化")
    
    def path_callback(self, msg):
        if len(msg.poses) > 0:
            self.path = msg
            self.current_waypoint_index = 0
            rospy.loginfo(f"接收到新路径，包含{len(msg.poses)}个路径点")
        else:
            rospy.logwarn("接收到空路径")
    
    def control_callback(self, event):
        if self.path is None or self.current_waypoint_index >= len(self.path.poses):
            # 没有路径可跟随或已到达终点
            self.stop_robot()
            return
        
        try:
            # 获取机器人当前位置
            self.tf_listener.waitForTransform("/map", self.robot_frame, rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform("/map", self.robot_frame, rospy.Time(0))
            
            robot_x, robot_y = trans[0], trans[1]
            (roll, pitch, yaw) = euler_from_quaternion(rot)
            
            # 获取当前目标路径点
            current_waypoint = self.path.poses[self.current_waypoint_index]
            target_x = current_waypoint.pose.position.x
            target_y = current_waypoint.pose.position.y
            
            # 发布当前目标用于可视化
            self.current_target_pub.publish(current_waypoint)
            
            # 计算到路径点的距离
            distance = np.sqrt((target_x - robot_x)**2 + (target_y - robot_y)**2)
            
            # 检查是否已到达路径点
            if distance < self.waypoint_reach_distance:
                self.current_waypoint_index += 1
                rospy.loginfo(f"已到达路径点{self.current_waypoint_index}，移动到下一个")
                if self.current_waypoint_index >= len(self.path.poses):
                    rospy.loginfo("已到达最终路径点！")
                    self.stop_robot()
                    return
            
            # 计算到目标的方向
            angle_to_target = np.arctan2(target_y - robot_y, target_x - robot_x)
            angle_error = angle_to_target - yaw
            
            # 将角度误差归一化到[-pi, pi]
            while angle_error > np.pi:
                angle_error -= 2*np.pi
            while angle_error < -np.pi:
                angle_error += 2*np.pi
            
            # 创建控制命令
            cmd = Twist()
            
            # 如果角度误差较大，首先原地旋转
            if abs(angle_error) > 0.3:
                cmd.angular.z = self.angular_velocity if angle_error > 0 else -self.angular_velocity
                cmd.linear.x = 0.0
            else:
                # 否则，前进并调整方向
                cmd.linear.x = self.linear_velocity
                cmd.angular.z = 0.5 * angle_error  # 比例控制
            
            self.cmd_vel_pub.publish(cmd)
            
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF错误: {str(e)}")
            self.stop_robot()
    
    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

if __name__ == '__main__':
    try:
        follower = PathFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
