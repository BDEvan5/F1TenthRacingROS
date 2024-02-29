# F1TenthRacingROS

The ROS2 package is used to test autonomous racing methods in the [ROS2 simulation](https://github.com/f1tenth/f1tenth_gym_ros) and onboard a [physical vehicle](f1tenth.com).
This code is not maintained.

The code is structured to have a resuable base (DriveNode) that can be subclassed for a specific planning method. This removes the need for the vehicle interface of subscribers and published to be reimplemented for every planning method.
The code was used with the pure pursuit planner and to test deep reinforcement learning agents with different architectures. Some of the results were published in our article, ["**Comparing deep reinforcement learning architectures for autonomous racing**"](https://www.sciencedirect.com/science/article/pii/S266682702300049X).

The code used to train the agents is available in the [f1tenth_drl](https://github.com/BDEvan5/f1tenth_drl) repo.

![trajectories](https://github.com/BDEvan5/F1TenthRacingROS/assets/31577482/dad70912-cff5-4d0f-835c-d4e9a98714ee)

If the code is useful to you, please consider citing us,
```latex
@article{evans2023comparing,
    title={Comparing deep reinforcement learning architectures for autonomous racing},
    author={Evans, Benjamin David and Jordaan, Hendrik Willem and Engelbrecht, Herman Arnold},
    journal={Machine Learning with Applications},
    volume={14},
    pages={100496},
    year={2023},
    publisher={Elsevier}
}
```

