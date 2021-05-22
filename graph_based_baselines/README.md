# Graph-based baselines

Graph-based baseline models can directly output graphs representing the road boundary. They all follow a similar pipeline: starting from predicted or human-defined initial vertex, iteratively generate the graph vertex by vertex until stop action is triggered. The process can be treated as a sequential graph growing process. It is also similar to visual navigation tasks in the robotics field, but with a different environment (drive an agent on the aerial image along the road boundary).

There are 6 graph-based methods implemented, and enhance-iCurb is the improved version of our previous work iCurb, in which the graph iterative generation is analyzed from the perspective of imitation learning.

## Advantages
* Topology aware. With different graph generate (or saying agent navigation) policy, these methods can effectively capture topological information and directly predict the graph of target objects.
* Post-processing is not required. Since the output is directly graphs, there is no need to convert pixel-level results to graphs.
* User-friendly. Since the process of generating the graph is very similar to that of how the object is annotated by human users, the output is more user-friendly for end-users. Users can understand and edit the generated graph without much trouble.

## Weaknesses
* Time inefficiency. The iterative graph generation is a sequential process, which takes huge time and resource consumption.
* Error accumulation. Vertex prediction heavily relies on the prediction of previous vertices. Tiny errors may accumulate and the agent may gradually get far from the right track.
* Instability. The graph generation process is very sensitive to changes in the overall pipeline, which may make it difficult to converge or severely degrade the final performance.

Despite the weaknesses, this category of methods receives more and more attention recently due to its excellent performance on topology correctness. The weaknesses could be relieved by proposing better training strategies and pipelines.

## Note
* At this stage, the graph-based baselines are not implemented with multi-GPU or multi-process to accelerate, which may be done in the future.
* For a fair comparison, ground-truth initial vertices are used to start the iterative graph generation.