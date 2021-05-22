# Segmentation-based baselines

Topo-boundary implements 3 segmentation-based baseline models, including 1 self-proposed naive baseline and 2 baselines based on previous works. All methods share a similar pipeline: first, predict the segmentation map of road boundaries, then utilize hard-code post-processing or other algorithms to refine the segmentation map.



## Advantages 
The main advantage of this category of methods are:
* Time efficiency. Since they directly work on pixels, parallelism is easy to realize.
* Stability. The experiments are easy to reproduce, and methods are not sensitive to minor changes in the model.
* No error accumulation. Not like graph-based baseline models. this category of methods is free from accumulated errors since each pixel prediction is independent.

## Weakness
* Unaware of topology. Since the segmentation is pixel-level prediction, it cannot effectively capture the relationship between pixels and output the correct topology. Therefore, the segmentation map usually has many mistaken disconnections or ghost connections.
* Hard to obtain graphs. Converting the pixel-level segmentation to graphs with vertices and edges can be quite hard and tedious.
* Not editable. Since the segmentation map is pixel-level, it cannot be edited by end-users like graphs. The output is of little value from the view of Human-Computer-Interaction.

Due to the importance of the topology of the target line-shaped object, most recent works turn to graph-based methods.