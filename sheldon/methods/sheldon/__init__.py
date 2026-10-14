"""SHELDON: dual search over a hypothesis space and an observation space.

Each round runs Explore, CandidateAnalysis (a build-and-evaluate loop per
hypothesis), ComparativeAnalysis (evidence that tells them apart) and Exploit
(retain / repair / refine / set aside).

Importing this package registers the method as `sheldon`.
"""

from sheldon.methods.sheldon.loop import Sheldon

__all__ = ["Sheldon"]
