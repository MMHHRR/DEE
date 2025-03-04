"""
管理多智能体之间的社交互动模块。
"""

import random
from datetime import datetime, timedelta
from utils import calculate_distance, parse_time, format_time_after_minutes

class SocialInteraction:
    def __init__(self):
        """初始化社交互动管理器"""
        self.social_networks = {}  # 存储每个智能体的社交网络
        self.planned_interactions = {}  # 存储计划中的互动
        self.interaction_history = {}  # 存储历史互动记录
        
    def initialize_social_network(self, persona_id, demographic_data):
        """
        为智能体初始化社交网络。
        
        Args:
            persona_id: 智能体ID
            demographic_data: 人口统计学数据
        """
        if persona_id not in self.social_networks:
            self.social_networks[persona_id] = {
                'family': set(),
                'friends': set(),
                'colleagues': set(),
                'demographic_data': demographic_data
            }
            
    def add_relationship(self, persona1_id, persona2_id, relationship_type):
        """
        添加两个智能体之间的关系。
        
        Args:
            persona1_id: 第一个智能体的ID
            persona2_id: 第二个智能体的ID
            relationship_type: 关系类型（family/friends/colleagues）
        """
        if persona1_id in self.social_networks and persona2_id in self.social_networks:
            self.social_networks[persona1_id][relationship_type].add(persona2_id)
            self.social_networks[persona2_id][relationship_type].add(persona1_id)
            
    def find_potential_interactions(self, persona_id, date, activities):
        """
        为特定智能体寻找潜在的社交互动机会。
        
        Args:
            persona_id: 智能体ID
            date: 日期
            activities: 当天的活动列表
            
        Returns:
            list: 可能的社交互动机会列表
        """
        potential_interactions = []
        
        if persona_id not in self.social_networks:
            return potential_interactions
            
        # 获取该智能体的所有社交联系人
        all_contacts = set()
        for relationship_type in ['family', 'friends', 'colleagues']:
            all_contacts.update(self.social_networks[persona_id][relationship_type])
            
        # 检查每个活动是否适合社交互动
        for activity in activities:
            activity_type = activity.get('activity_type', '').lower()
            
            # 某些活动类型更适合社交互动
            if activity_type in ['leisure', 'dining', 'recreation', 'social']:
                start_time = activity.get('start_time')
                end_time = activity.get('end_time')
                location = activity.get('location')
                
                # 寻找可能参与互动的联系人
                for contact_id in all_contacts:
                    if self._can_interact(persona_id, contact_id, date, start_time, end_time, location):
                        potential_interactions.append({
                            'activity': activity,
                            'contact_id': contact_id,
                            'start_time': start_time,
                            'end_time': end_time,
                            'location': location
                        })
                        
        return potential_interactions
        
    def _can_interact(self, persona1_id, persona2_id, date, start_time, end_time, location):
        """
        检查两个智能体是否可以在给定时间和地点进行互动。
        
        Args:
            persona1_id: 第一个智能体的ID
            persona2_id: 第二个智能体的ID
            date: 日期
            start_time: 开始时间
            end_time: 结束时间
            location: 位置
            
        Returns:
            bool: 是否可以互动
        """
        # 检查是否已经有计划的互动
        if (date, persona1_id) in self.planned_interactions or (date, persona2_id) in self.planned_interactions:
            existing_interaction = self.planned_interactions.get((date, persona1_id)) or \
                                 self.planned_interactions.get((date, persona2_id))
            
            if existing_interaction:
                # 检查时间是否重叠
                existing_start = existing_interaction['start_time']
                existing_end = existing_interaction['end_time']
                
                if (parse_time(start_time) < parse_time(existing_end) and 
                    parse_time(end_time) > parse_time(existing_start)):
                    return False
                    
        # 根据关系类型和时间段调整互动概率
        relationship_type = self._get_relationship_type(persona1_id, persona2_id)
        if not relationship_type:
            return False
            
        # 计算互动概率
        base_probability = {
            'family': 0.7,
            'friends': 0.5,
            'colleagues': 0.3
        }.get(relationship_type, 0.1)
        
        # 根据时间调整概率
        hour = int(start_time.split(':')[0])
        if 9 <= hour <= 17:  # 工作时间
            base_probability *= 0.5
        elif 18 <= hour <= 22:  # 晚上
            base_probability *= 1.5
            
        return random.random() < base_probability
        
    def _get_relationship_type(self, persona1_id, persona2_id):
        """
        获取两个智能体之间的关系类型。
        
        Args:
            persona1_id: 第一个智能体的ID
            persona2_id: 第二个智能体的ID
            
        Returns:
            str: 关系类型
        """
        if persona1_id not in self.social_networks or persona2_id not in self.social_networks:
            return None
            
        for relationship_type in ['family', 'friends', 'colleagues']:
            if (persona2_id in self.social_networks[persona1_id][relationship_type] or
                persona1_id in self.social_networks[persona2_id][relationship_type]):
                return relationship_type
                
        return None
        
    def plan_interaction(self, persona1_id, persona2_id, date, activity):
        """
        规划两个智能体之间的互动。
        
        Args:
            persona1_id: 第一个智能体的ID
            persona2_id: 第二个智能体的ID
            date: 日期
            activity: 活动详情
        """
        interaction = {
            'participants': [persona1_id, persona2_id],
            'date': date,
            'start_time': activity['start_time'],
            'end_time': activity['end_time'],
            'location': activity['location'],
            'activity_type': activity['activity_type']
        }
        
        # 记录计划的互动
        self.planned_interactions[(date, persona1_id)] = interaction
        self.planned_interactions[(date, persona2_id)] = interaction
        
    def record_interaction(self, date, interaction):
        """
        记录已发生的互动。
        
        Args:
            date: 日期
            interaction: 互动详情
        """
        if date not in self.interaction_history:
            self.interaction_history[date] = []
        self.interaction_history[date].append(interaction)
        
    def get_interaction_history(self, persona_id, date_range=None):
        """
        获取智能体的互动历史。
        
        Args:
            persona_id: 智能体ID
            date_range: 日期范围（可选）
            
        Returns:
            list: 互动历史记录
        """
        history = []
        for date, interactions in self.interaction_history.items():
            if date_range and date not in date_range:
                continue
            for interaction in interactions:
                if persona_id in interaction['participants']:
                    history.append(interaction)
        return history 