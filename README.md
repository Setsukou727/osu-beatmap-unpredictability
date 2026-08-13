# osu! Beatmap Unpredictability

An experimental approach to measuring the unpredictability and pattern diversity of osu! beatmaps.

>Note: The core idea and algorithm design of this project are my own. However, since I am not a native English speaker and this is my first time releasing an open-source project, the code implementation and this README were heavily assisted by AI.

## Motivation

osu! difficulty and performance systems have been continuously updated, but one problem remains difficult to capture consistently: some tech-oriented beatmaps can feel harder than their calculated difficulty suggests, while some farm-oriented beatmaps can receive comparatively high difficulty estimates.

This project explores the hypothesis that an important difference between these maps is **unpredictability**.

When playing a beatmap, the player does not simply react to isolated objects. Previous experience allows them to recognize patterns and anticipate where subsequent objects are likely to appear. In this sense, a pattern that closely resembles familiar patterns should be easier to predict, while a beatmap that contains a larger variety of distinct patterns may require a broader collection of learned patterns from the player.

This leads to the following hypothesis:

> The more diverse the spatial-temporal patterns required by a beatmap, the less predictable the beatmap becomes.

The purpose of this project is to explore whether this idea can be quantified and whether such a measure could eventually become an additional dimension of osu! difficulty, or potentially contribute to the reading-related aspects of a difficulty model.

## Approach

The beatmap is divided into windows of different musical lengths. Each window contains a sequence of objects represented in three-dimensional space-time:

[
(x, y, t)
]

where `x` and `y` describe the object's position and `t` describes its position in musical time.

### Representing a pattern

A pattern can be regarded as a geometric configuration of points in this three-dimensional space.

For a finite set of points in Euclidean space, the collection of pairwise distances determines its geometric structure up to transformations such as translation, rotation, and reflection. These transformations are intentionally ignored here because they do not fundamentally change the relative arrangement of the objects.

Therefore, instead of directly comparing absolute coordinates, the implementation constructs a **distance signature** from the pairwise distances between objects within a window.

This provides a representation of a pattern that focuses on its internal structure rather than its absolute position or orientation.

### Comparing patterns

Patterns with similar distance signatures are treated as similar spatial-temporal arrangements.

The implementation compares repeated occurrences of the same symbolic pattern and estimates their continuous geometric similarity. Temporal proximity is also taken into account, so recently encountered patterns have a stronger influence than patterns encountered much earlier.

The resulting local value represents how different an occurrence is from other occurrences of the same symbolic pattern.

### From local variation to global unpredictability

The local complexity values are combined across multiple window sizes.

The final score is intended to represent the overall diversity of the patterns used by a beatmap:

* repeated and geometrically similar patterns contribute less;
* unusual variations contribute more;
* patterns occurring at different musical scales can contribute differently.

The resulting value is referred to as the beatmap's **unpredictability**.

## Current Status

This repository contains an **experimental prototype**, not a finished difficulty or performance algorithm.

The current implementation can calculate an unpredictability score, but its results do not yet consistently match the intended behaviour.

In particular, some jump-oriented farm maps can still receive unexpectedly high unpredictability scores. The current model therefore should not be interpreted as a reliable measure of beatmap difficulty.

The main purpose of this project is to make the underlying idea and its initial implementation available for experimentation, criticism, and further development.

## Usage

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then run the program with a beatmap file or directory:

```bash
python unpredictability.py <path>
```

The program accepts `.osu` and `.osz` files, as well as directories containing beatmaps.

For example:

```bash
python unpredictability.py maps/
```

The program outputs an unpredictability score for each successfully analyzed beatmap.

## Limitations

The current implementation has several limitations.

The score has not been calibrated against player performance, existing difficulty attributes, or osu! pp values. No claim is made that the current score corresponds directly to perceived difficulty.

The current method also depends on several heuristic choices, including:

* the lengths of the analysis windows;
* symbolic quantization of rhythm;
* the relative weighting of spatial and temporal distance;
* temporal memory;
* the weighting of different analysis scales;
* the aggregation of local pattern complexity into a global score.

These choices are still experimental.

The current implementation also does not fully explain why certain jump-oriented patterns receive high scores, which is an important open problem.

## Open Questions

Several aspects of the model remain open for further investigation:

* What is the most appropriate definition of a pattern window?
* How should spatial and temporal differences be balanced?
* How should repeated patterns affect unpredictability?
* How should short and long patterns be weighted?
* How should nested or overlapping patterns be handled?
* Why do some jump-oriented maps still receive high unpredictability scores?
* How should an unpredictability measure interact with existing difficulty attributes?
* Is pairwise geometric similarity the appropriate mathematical framework for this problem?

Alternative mathematical formulations are also welcome.

## Contributing

This project is intentionally open to further development.

Suggestions, criticism, experiments, alternative mathematical formulations, and improved implementations are welcome. The current implementation is best regarded as a starting point for exploring the underlying idea rather than as a final solution.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
