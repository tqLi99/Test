#!/usr/bin/env python3
import rospy
import requests
import json
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np
import os
from openai import OpenAI

class LLMPathPlanner:
    def __init__(self):
        rospy.init_node('llm_path_planner')
        
        # 参数设置
        self.llm_api_url = rospy.get_param('~llm_api_url', 'http://localhost:8000/generate')
        self.api_key = rospy.get_param('~api_key', '')
        self.grid_resolution = rospy.get_param('~grid_resolution', 0.1)  # 米
        self.map_width = rospy.get_param('~map_width', 10)  # 米
        self.map_height = rospy.get_param('~map_height', 10)  # 米
        
        # 机器人当前位置（默认在原点）
        self.robot_position = (0.0, 0.0)
        
        # 目标位置
        self.target_position = (0.0, 0.0)
        self.target_set = False
        
        # 障碍物信息
        self.obstacle_info = "未检测到障碍物。"
        
        # 发布器
        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=1)
        self.path_text_pub = rospy.Publisher('/path_text', String, queue_size=1)
        
        # 订阅器
        rospy.Subscriber('/obstacle_info', String, self.obstacle_callback)
        rospy.Subscriber('/initialpose', PoseStamped, self.robot_pose_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        
        rospy.loginfo("LLM路径规划器已初始化")
        rospy.Timer(rospy.Duration(5), self.plan_path_timer_callback)
    
    def obstacle_callback(self, msg):
        self.obstacle_info = msg.data
        rospy.loginfo("已接收障碍物信息")
    
    def robot_pose_callback(self, msg):
        self.robot_position = (msg.pose.position.x, msg.pose.position.y)
        rospy.loginfo(f"机器人位置已设置为: {self.robot_position}")
    
    def goal_callback(self, msg):
        self.target_position = (msg.pose.position.x, msg.pose.position.y)
        self.target_set = True
        rospy.loginfo(f"目标位置已设置为: {self.target_position}")
        self.plan_path()
    
    def plan_path_timer_callback(self, event):
        if self.target_set:
            self.plan_path()
    
    def plan_path(self):
        if not self.target_set:
            rospy.logwarn("目标位置未设置，无法规划路径")
            return
        
        prompt = self.create_llm_prompt()
        path_text = self.query_llm(prompt)
        
        if path_text:
            self.path_text_pub.publish(String(path_text))
            path = self.parse_path_from_llm(path_text)
            if path:
                self.publish_path(path)
    
    def create_llm_prompt(self):
     prompt = f"""
作为路径规划专家，请根据以下信息为机器人生成一条从当前位置到目标位置的安全路径。

环境信息:
- 地图大小: {self.map_width}x{self.map_height}米
- 机器人当前位置: ({self.robot_position[0]:.2f}, {self.robot_position[1]:.2f})
- 目标位置: ({self.target_position[0]:.2f}, {self.target_position[1]:.2f})

障碍物信息:
{self.obstacle_info}

请注意：
1. 路径必须避开所有障碍物，与障碍物保持至少0.3米的安全距离
2. 路径应尽可能短且平滑
3. 相邻路径点之间的距离不应超过0.5米
4. 转弯处需要足够的路径点确保平滑过渡

请严格按照以下格式返回路径点序列:
PATH:
x1,y1
x2,y2
...
xn,yn
END_PATH

请仅生成符合上述格式的路径点，不要包含任何说明或解释。
"""
     return prompt
    
    def query_llm(self, prompt):
    try:
        import os
        from openai import OpenAI
        
        # 创建客户端
        client = OpenAI(
            # 如果设置了环境变量，使用环境变量；否则使用参数中的api_key
            api_key=os.getenv("DASHSCOPE_API_KEY") or self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 调用API
        completion = client.chat.completions.create(
            model="deepseek-r1",  # 也可以使用其他百炼支持的模型
            messages=[
                {"role": "system", "content": "你是一个专业的路径规划专家，擅长为机器人生成安全、高效的运动路径。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # 降低温度以获得更确定性的结果
            max_tokens=1000
        )
        
        # 返回生成的内容
        if completion and completion.choices and len(completion.choices) > 0:
            return completion.choices[0].message.content
        else:
            rospy.logerr("未从API获取到有效响应")
            return None
            
    except Exception as e:
        rospy.logerr(f"查询LLM时出错: {str(e)}")
        return None
    
    def parse_path_from_llm(self, llm_response):
        try:
            lines = llm_response.strip().split('\n')
            path_start_index = -1
            path_end_index = -1
            
            for i, line in enumerate(lines):
                if 'PATH:' in line:
                    path_start_index = i + 1
                if 'END_PATH' in line:
                    path_end_index = i
            
            if path_start_index == -1 or path_end_index == -1 or path_start_index >= path_end_index:
                rospy.logerr("无法从LLM响应中解析路径")
                return None
            
            path_lines = lines[path_start_index:path_end_index]
            path = []
            
            for line in path_lines:
                if ',' in line:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            path.append((x, y))
                        except ValueError:
                            rospy.logwarn(f"无法从行中解析坐标: {line}")
            
            return path
        
        except Exception as e:
            rospy.logerr(f"从LLM响应中解析路径时出错: {str(e)}")
            return None
    
    def publish_path(self, path_points):
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"
        
        for x, y in path_points:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
        rospy.loginfo(f"已发布包含{len(path_points)}个路径点的路径")

if __name__ == '__main__':
    try:
        planner = LLMPathPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
        
