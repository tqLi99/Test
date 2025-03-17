#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import String

class LidarObstacleDetector:
    def __init__(self):
        rospy.init_node('lidar_obstacle_detector')
        
        # 参数设置
        self.distance_threshold = rospy.get_param('~distance_threshold', 1.0)  # 米
        self.cluster_threshold = rospy.get_param('~cluster_threshold', 0.3)    # 米
        self.min_points_per_obstacle = rospy.get_param('~min_points_per_obstacle', 3)
        
        # 发布器
        self.obstacle_marker_pub = rospy.Publisher('/obstacle_markers', MarkerArray, queue_size=1)
        self.obstacle_info_pub = rospy.Publisher('/obstacle_info', String, queue_size=1)
        
        # 订阅器
        rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        
        self.obstacles = []
        rospy.loginfo("激光雷达障碍物探测器已初始化")
    
    def laser_callback(self, scan_msg):
        # 将激光扫描转换为点
        points = self.laser_scan_to_points(scan_msg)
        
        # 从点中检测障碍物
        self.obstacles = self.detect_obstacles(points)
        
        # 发布可视化标记
        self.publish_obstacle_markers()
        
        # 发布障碍物信息文本
        self.publish_obstacle_info()
    
    def laser_scan_to_points(self, scan_msg):
        points = []
        angle = scan_msg.angle_min
        
        for r in scan_msg.ranges:
            if scan_msg.range_min <= r <= scan_msg.range_max:
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                if r < self.distance_threshold:  # 只考虑阈值范围内的点
                    points.append((x, y))
            angle += scan_msg.angle_increment
        
        return points
    
    def detect_obstacles(self, points):
        if not points:
            return []
        
        obstacles = []
        current_cluster = []
        
        # 简单的聚类算法
        sorted_points = sorted(points)
        current_cluster.append(sorted_points[0])
        
        for i in range(1, len(sorted_points)):
            prev_point = sorted_points[i-1]
            current_point = sorted_points[i]
            
            distance = np.sqrt((current_point[0] - prev_point[0])**2 + 
                              (current_point[1] - prev_point[1])**2)
            
            if distance < self.cluster_threshold:
                current_cluster.append(current_point)
            else:
                if len(current_cluster) >= self.min_points_per_obstacle:
                    # 计算障碍物中心
                    center_x = sum(p[0] for p in current_cluster) / len(current_cluster)
                    center_y = sum(p[1] for p in current_cluster) / len(current_cluster)
                    
                    # 估计大小
                    max_dist = 0
                    for p in current_cluster:
                        dist = np.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2)
                        max_dist = max(max_dist, dist)
                    
                    obstacles.append({
                        'center': (center_x, center_y),
                        'radius': max_dist,
                        'points': len(current_cluster)
                    })
                
                current_cluster = [current_point]
        
        # 检查最后一个聚类
        if len(current_cluster) >= self.min_points_per_obstacle:
            center_x = sum(p[0] for p in current_cluster) / len(current_cluster)
            center_y = sum(p[1] for p in current_cluster) / len(current_cluster)
            
            max_dist = 0
            for p in current_cluster:
                dist = np.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2)
                max_dist = max(max_dist, dist)
            
            obstacles.append({
                'center': (center_x, center_y),
                'radius': max_dist,
                'points': len(current_cluster)
            })
        
        return obstacles
    
    def publish_obstacle_markers(self):
        marker_array = MarkerArray()
        
        for i, obstacle in enumerate(self.obstacles):
            marker = Marker()
            marker.header.frame_id = "laser"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            marker.pose.position.x = obstacle['center'][0]
            marker.pose.position.y = obstacle['center'][1]
            marker.pose.position.z = 0
            
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = obstacle['radius'] * 2
            marker.scale.y = obstacle['radius'] * 2
            marker.scale.z = 0.1
            
            marker.color.a = 0.7
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            
            marker_array.markers.append(marker)
        
        self.obstacle_marker_pub.publish(marker_array)
    
    def publish_obstacle_info(self):
        if not self.obstacles:
            obstacle_text = "未检测到障碍物。"
        else:
            obstacle_text = f"检测到{len(self.obstacles)}个障碍物:\n"
            for i, obs in enumerate(self.obstacles):
                obstacle_text += f"障碍物{i+1}: 位置({obs['center'][0]:.2f}, {obs['center'][1]:.2f})，半径{obs['radius']:.2f}米\n"
        
        self.obstacle_info_pub.publish(String(obstacle_text))

if __name__ == '__main__':
    try:
        detector = LidarObstacleDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
