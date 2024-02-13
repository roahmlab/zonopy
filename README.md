# zonopy - Zonotopes in Python

Base implementation of various set representations in Python.
See documentation [here](https://roahmlab.github.io/zonopy).

Much of the math behind the set types implemented here extend from [CORA](https://tumcps.github.io/CORA/).

## Continuous sets types implemented

* Interval
* Zonotopes
* Polynomial Zonotopes

## Installation instructions

Either clone the latest version or a specific tag of this repo and inside of the repo path, run:
```
pip install -e .
```

## Prerequisites

```
pytorch >= 2.0.0
numpy >= 1.24
scipy >= 1.9.0
```

## How to Cite

If you use zonopy or zonopy-robots in your research, please cite one or more of the following papers:

<!-- Safe Planning for Articulated Robots Using Reachability-based Obstacle Avoidance With Spheres. J. Michaux, A. Li, Q. Chen, C. Chen, B. Zhang, and R. Vasudevan. ArXiv, 2024. ()
```bibtex
@article{michaux2024sparrows,
  title={Safe Planning for Articulated Robots Using Reachability-based Obstacle Avoidance With Spheres},
  author={Jonathan Michaux and Adam Li and Qingyi Chen and Che Chen and Bohao Zhang and Ram Vasudevan},
  journal={ArXiv},
  year={2024},
  volume={},
}
``` -->

Reachability-based Trajectory Design with Neural Implicit Safety Constraints. J. B. Michaux, Y. S. Kwon, Q. Chen, and R. Vasudevan. Robotics: Science and Systems, 2023. (https://www.roboticsproceedings.org/rss19/p062.pdf)
```bibtex
@inproceedings{michaux2023rdf,
  title={{Reachability-based Trajectory Design with Neural Implicit Safety Constraints}},
  author={Jonathan B Michaux AND Yong Seok Kwon AND Qingyi Chen AND Ram Vasudevan},
  booktitle={Proceedings of Robotics: Science and Systems},
  year={2023},
  address={Daegu, Republic of Korea},
  doi={10.15607/RSS.2023.XIX.062}
}
```
