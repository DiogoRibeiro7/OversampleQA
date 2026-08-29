# Algorithms

## Probability Distance Metrics

### Hellinger Distance

The Hellinger distance between two discrete probability vectors $`p`$
and $`q`$ is defined as:

``` math
H(p, q) = \frac{1}{\sqrt{2}} \left\| \sqrt{p} - \sqrt{q} \right\|_2
```

This metric ranges between 0 and 1 and measures the similarity between
probability distributions.

### Jensen--Shannon Distance

The Jensen--Shannon distance derives from the Jensen--Shannon
divergence:

<span nowrap="">
``` math
\begin{aligned}
\begin{align}
JSD(p, q) &= \frac{1}{2} KL(p \parallel m) + \frac{1}{2} KL(q \parallel m)\\
m &= \frac{1}{2}(p + q)
\end{align}
\end{aligned}
```
</span>

``` math
JSDist(p, q) = \sqrt{JSD(p, q)}
```

This symmetric measure is always finite and lies between 0 and 1.

### Hassanat Distance

For real-valued vectors $`x`$ and $`y`$, with $`m = \min(x_i, y_i)`$ and
$`M = \max(x_i, y_i)`$, the Hassanat distance computes per-dimension
contributions:

``` math
\begin{aligned}
d_i = \begin{cases}
    1 - \dfrac{1 + m}{1 + M} & \text{if } m \ge 0\\[2ex]
    1 - \dfrac{1 + m + |m|}{1 + M + |m|} & \text{if } m < 0
\end{cases}
\end{aligned}
```

The overall distance is the sum $`\sum_i d_i`$.

The unit shift in numerator and denominator is essential. It bounds
every per-dimension contribution to $`[0, 1)`$, which is what makes the
metric invariant to feature scale and robust to outliers: no single
dimension can dominate the sum regardless of its magnitude. The shift
also keeps the denominator $`\ge 1`$, so the expression is well defined
everywhere, including at the origin.

Because the contribution is bounded, the metric is well suited to
unnormalised features with very different scales — a dimension measured
in millions contributes no more than a dimension measured in fractions.

See `bibliography` for full references.
