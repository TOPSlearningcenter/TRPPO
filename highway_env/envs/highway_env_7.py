from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray



class HighwayEnv_7(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "absolute": False,
                    "normalize": False,
                    "vehicles_count": 15,
                    "see_behind": True,
                },
                "action": {
                    "type": "DiscreteMetaAction",
                    "target_speeds": np.linspace(5, 32, 9),
                },
                "lanes_count": 3,
                "vehicles_count": 30,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 30,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -50,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 5,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "Headway_time":1.2,
                "Headway_cost":0,
                "Low_speed_cost": 20,
                "offroad_terminal": False,
                "screen_width": 1500,  # [px]
                "screen_height": 150,  # [px]
                "scaling": 4,
                "seed": 7
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        # self.road = Road(
        #     network=RoadNetwork.straight_road_network(
        #         self.config["lanes_count"], speed_limit=30
        #     ),
        #     np_random=self.np_random,
        #     record_history=self.config["show_trajectories"],
        # )
        net = RoadNetwork()

        # Highway lanes
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        net.add_lane("a", "b", StraightLane([0, 0], [5000,0], line_types=[c, s]))
        net.add_lane("a", "c", StraightLane([0, StraightLane.DEFAULT_WIDTH], [5000, StraightLane.DEFAULT_WIDTH], line_types=[n, s]))
        net.add_lane("a", "d", StraightLane([0, 2 * StraightLane.DEFAULT_WIDTH], [5000, 2 * StraightLane.DEFAULT_WIDTH], line_types=[n, c]))

        road = Road(network=net, record_history=self.config["show_trajectories"])
        self.road = road
    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []

        spawn_points_s1 = [10, 50, 90, 130, 170, 210]
        spawn_points_s2 = [5, 45, 85, 125, 165, 205]
        spawn_points_s3 = [15, 55, 95, 135, 175, 215]

        spawn_point_s_c = np.random.choice(spawn_points_s2, replace=False)
        spawn_point_s_c = [spawn_point_s_c]
        spawn_point_s_c = [125]
        for a in spawn_point_s_c:
            spawn_points_s2.remove(a)

        spawn_point_s_h1 = np.random.choice(spawn_points_s1, 3, replace=False)
        spawn_point_s_h2 = np.random.choice(spawn_points_s2, 3, replace=False)
        spawn_point_s_h3 = np.random.choice(spawn_points_s3, 3, replace=False)

        # spawn_point_s_h1 = [90, 170, 210]
        # spawn_point_s_h2 = [85, 165, 205]
        # spawn_point_s_h3 = [55, 135, 175]

        spawn_point_s_h1 = list(spawn_point_s_h1)
        spawn_point_s_h2 = list(spawn_point_s_h2)
        spawn_point_s_h3 = list(spawn_point_s_h3)

        initial_speed = (np.random.rand(1 + 9) * 2 + 25)
        loc_noise = (np.random.rand(1 + 9) * 3 - 1.5)
        initial_speed = list(initial_speed)
        loc_noise = list(loc_noise)

        for _ in range(1):
            ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("a", "c", 0)).position(
                spawn_point_s_c.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0))
            self.controlled_vehicles.append(ego_vehicle)
            road.vehicles.append(ego_vehicle)

        for _ in range(3):
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(
                    spawn_point_s_h1.pop(0) + loc_noise
                    .pop(0), 0), speed=initial_speed.pop(0)))

        for _ in range(3):
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("a", "c", 0)).position(
                    spawn_point_s_h2.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0)))

        for _ in range(3):
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("a", "d", 0)).position(
                    spawn_point_s_h3.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0)))

        # other_per_controlled = near_split(
        #     self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        # )
        #
        # self.controlled_vehicles = []
        # for others in other_per_controlled:
        #     vehicle = Vehicle.create_random(
        #         self.road,
        #         speed=25,
        #         lane_id=self.config["initial_lane_id"],
        #         spacing=self.config["ego_spacing"],
        #     )
        #     vehicle = self.action_type.vehicle_class(
        #         self.road, vehicle.position, vehicle.heading, vehicle.speed
        #     )
        #     self.controlled_vehicles.append(vehicle)
        #     self.road.vehicles.append(vehicle)
        #
        #     for _ in range(others):
        #         vehicle = other_vehicles_type.create_random(
        #             self.road, spacing=1 / self.config["vehicles_density"]
        #         )
        #         vehicle.randomize_behavior()
        #         self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [-1, 1]
        )

        low_speed_cost = np.log(self.vehicle.speed/20)

        headway_distance = self._compute_headway_distance(self.vehicle)
        Headway_cost = np.log(
            headway_distance / (self.config["Headway_time"] * self.vehicle.speed)
        )if self.vehicle.speed > 0 else 0
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
            "Headway_cost": float(Headway_cost if Headway_cost < 0 else 0),
            "Low_speed_cost": float(low_speed_cost),
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _compute_headway_distance(self, vehicle):
        headway_distance = 60
        for v in self.road.vehicles:
            if (v.lane_index == vehicle.lane_index) and (v.position[0] > vehicle.position[0]):
                hd = v.position[0] - vehicle.position[0]
                if hd < headway_distance:
                    headway_distance = hd

        return headway_distance
