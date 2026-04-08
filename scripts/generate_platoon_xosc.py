#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path


VEHICLE_MODEL = "vehicle.tesla.model3"


def _vehicle_object(name: str, color: str) -> str:
    return f"""    <ScenarioObject name="{name}">
      <Vehicle name="{VEHICLE_MODEL}" vehicleCategory="car">
        <Performance maxSpeed="69.444" maxAcceleration="10.0" maxDeceleration="10.0"/>
        <BoundingBox>
          <Center x="1.5" y="0.0" z="0.9"/>
          <Dimensions width="2.1" length="4.8" height="1.8"/>
        </BoundingBox>
        <Axles>
          <FrontAxle maxSteering="0.5" wheelDiameter="0.6" trackWidth="1.8" positionX="3.1" positionZ="0.3"/>
          <RearAxle maxSteering="0.0" wheelDiameter="0.6" trackWidth="1.8" positionX="0.0" positionZ="0.3"/>
        </Axles>
        <Properties>
          <Property name="type" value="simulation"/>
          <Property name="role_name" value="{name}"/>
          <Property name="ros_name" value="{name}"/>
          <Property name="color" value="{color}"/>
        </Properties>
      </Vehicle>
    </ScenarioObject>"""


def _teleport_action(name: str, x: float, y: float, z: float, h_rad: float) -> str:
    return f"""        <Private entityRef="{name}">
          <PrivateAction>
            <TeleportAction>
              <Position>
                <WorldPosition x="{x:.3f}" y="{y:.3f}" z="{z:.3f}" h="{h_rad:.6f}"/>
              </Position>
            </TeleportAction>
          </PrivateAction>
          <PrivateAction>
            <LongitudinalAction>
              <SpeedAction>
                <SpeedActionDynamics dynamicsShape="step" value="0.1" dynamicsDimension="time"/>
                <SpeedActionTarget>
                  <AbsoluteTargetSpeed value="0.0"/>
                </SpeedActionTarget>
              </SpeedAction>
            </LongitudinalAction>
          </PrivateAction>
        </Private>"""


def _speed_event(index: int, start_time: float, speed: float, accel: float, cruise: float, brake_target: float, decel: float) -> str:
    brake_time = start_time + max((speed - brake_target) / max(accel, 0.1), 0.1) + cruise
    return f"""              <Event name="LeaderAccelerate{index}" priority="overwrite">
                <Action name="LeaderAccelerate{index}">
                  <PrivateAction>
                    <LongitudinalAction>
                      <SpeedAction>
                        <SpeedActionDynamics dynamicsShape="linear" value="{accel:.3f}" dynamicsDimension="rate"/>
                        <SpeedActionTarget>
                          <AbsoluteTargetSpeed value="{speed:.3f}"/>
                        </SpeedActionTarget>
                      </SpeedAction>
                    </LongitudinalAction>
                  </PrivateAction>
                </Action>
                <StartTrigger>
                  <ConditionGroup>
                    <Condition name="StartAccelerate{index}" delay="0" conditionEdge="rising">
                      <ByValueCondition>
                        <SimulationTimeCondition value="{start_time:.3f}" rule="greaterThan"/>
                      </ByValueCondition>
                    </Condition>
                  </ConditionGroup>
                </StartTrigger>
              </Event>
              <Event name="LeaderBrake{index}" priority="overwrite">
                <Action name="LeaderBrake{index}">
                  <PrivateAction>
                    <LongitudinalAction>
                      <SpeedAction>
                        <SpeedActionDynamics dynamicsShape="linear" value="{decel:.3f}" dynamicsDimension="rate"/>
                        <SpeedActionTarget>
                          <AbsoluteTargetSpeed value="{brake_target:.3f}"/>
                        </SpeedActionTarget>
                      </SpeedAction>
                    </LongitudinalAction>
                  </PrivateAction>
                </Action>
                <StartTrigger>
                  <ConditionGroup>
                    <Condition name="StartBrake{index}" delay="0" conditionEdge="rising">
                      <ByValueCondition>
                        <SimulationTimeCondition value="{brake_time:.3f}" rule="greaterThan"/>
                      </ByValueCondition>
                    </Condition>
                  </ConditionGroup>
                </StartTrigger>
              </Event>"""


def build_scenario(args: argparse.Namespace) -> str:
    rng = random.Random(args.seed)
    heading = math.radians(args.yaw_deg)
    backward_x = -math.cos(heading)
    backward_y = -math.sin(heading)

    gaps = [rng.uniform(args.gap_min, args.gap_max) for _ in range(3)]
    positions = [0.0]
    for gap in gaps:
        positions.append(positions[-1] + gap)

    names = ["leader", "follower_1", "follower_2", "follower_3"]
    colors = ["255,0,0", "0,0,255", "0,128,255", "0,255,128"]
    objects = "\n".join(_vehicle_object(name, color) for name, color in zip(names, colors, strict=False))
    teleports = "\n".join(
        _teleport_action(
            name,
            args.x + backward_x * offset,
            args.y + backward_y * offset,
            args.z,
            heading,
        )
        for name, offset in zip(names, positions, strict=False)
    )

    events = []
    t = args.first_event_time
    for index in range(args.events):
        speed = rng.choice([15.0, 17.5, 20.0])
        accel = rng.uniform(args.accel_min, args.accel_max)
        cruise = rng.uniform(args.cruise_min, args.cruise_max)
        brake_target = rng.uniform(args.brake_target_min, args.brake_target_max)
        decel = rng.uniform(args.decel_min, args.decel_max)
        events.append(_speed_event(index + 1, t, speed, accel, cruise, brake_target, decel))
        stop_hold = rng.uniform(args.stop_hold_min, args.stop_hold_max)
        t += (
            max((speed - brake_target) / max(accel, 0.1), 0.1)
            + cruise
            + max((speed - brake_target) / max(decel, 0.1), 0.1)
            + stop_hold
        )

    event_xml = "\n".join(events)
    stop_time = max(t + 5.0, args.duration)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<OpenSCENARIO>
  <FileHeader revMajor="1" revMinor="0" date="2026-04-08T00:00:00" description="Platoon longitudinal repeated braking for RL" author="Platooning_with_CMDP"/>
  <CatalogLocations/>
  <RoadNetwork>
    <LogicFile filepath="{args.map}"/>
    <SceneGraphFile filepath=""/>
  </RoadNetwork>
  <Entities>
{objects}
  </Entities>
  <Storyboard>
    <Init>
      <Actions>
        <GlobalAction>
          <EnvironmentAction>
            <Environment name="ClearNoon">
              <TimeOfDay animation="false" dateTime="2026-04-08T12:00:00"/>
              <Weather cloudState="free">
                <Sun intensity="1.0" azimuth="0.0" elevation="1.31"/>
                <Fog visualRange="100000.0"/>
                <Precipitation precipitationType="dry" intensity="0.0"/>
              </Weather>
              <RoadCondition frictionScaleFactor="1.0"/>
            </Environment>
          </EnvironmentAction>
        </GlobalAction>
{teleports}
      </Actions>
    </Init>
    <Story name="PlatoonStory">
      <Act name="RepeatedBrakeAct">
        <ManeuverGroup maximumExecutionCount="1" name="LeaderEvents">
          <Actors selectTriggeringEntities="false">
            <EntityRef entityRef="leader"/>
          </Actors>
          <Maneuver name="LeaderVelocityManeuver">
{event_xml}
          </Maneuver>
        </ManeuverGroup>
        <StartTrigger>
          <ConditionGroup>
            <Condition name="ActStart" delay="0" conditionEdge="rising">
              <ByValueCondition>
                <SimulationTimeCondition value="0.1" rule="greaterThan"/>
              </ByValueCondition>
            </Condition>
          </ConditionGroup>
        </StartTrigger>
        <StopTrigger>
          <ConditionGroup>
            <Condition name="ActStop" delay="0" conditionEdge="rising">
              <ByValueCondition>
                <SimulationTimeCondition value="{stop_time:.3f}" rule="greaterThan"/>
              </ByValueCondition>
            </Condition>
          </ConditionGroup>
        </StopTrigger>
      </Act>
    </Story>
    <StopTrigger>
      <ConditionGroup>
        <Condition name="ScenarioStop" delay="0" conditionEdge="rising">
          <ByValueCondition>
            <SimulationTimeCondition value="{stop_time:.3f}" rule="greaterThan"/>
          </ByValueCondition>
        </Condition>
      </ConditionGroup>
    </StopTrigger>
  </Storyboard>
</OpenSCENARIO>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="scenarios/generated/platoon_longitudinal_brake.xosc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map", default="Town04")
    parser.add_argument("--x", type=float, default=392.68)
    parser.add_argument("--y", type=float, default=-187.71)
    parser.add_argument("--z", type=float, default=0.2)
    parser.add_argument("--yaw-deg", type=float, default=90.6)
    parser.add_argument("--gap-min", type=float, default=8.0)
    parser.add_argument("--gap-max", type=float, default=12.0)
    parser.add_argument("--events", type=int, default=5)
    parser.add_argument("--duration", type=float, default=100.0)
    parser.add_argument("--first-event-time", type=float, default=0.5)
    parser.add_argument("--cruise-min", type=float, default=4.0)
    parser.add_argument("--cruise-max", type=float, default=11.0)
    parser.add_argument("--accel-min", type=float, default=0.8)
    parser.add_argument("--accel-max", type=float, default=1.4)
    parser.add_argument("--decel-min", type=float, default=3.0)
    parser.add_argument("--decel-max", type=float, default=6.0)
    parser.add_argument("--brake-target-min", type=float, default=0.0)
    parser.add_argument("--brake-target-max", type=float, default=0.0)
    parser.add_argument("--stop-hold-min", type=float, default=3.0)
    parser.add_argument("--stop-hold-max", type=float, default=7.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_scenario(args), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
