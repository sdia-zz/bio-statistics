## Sources

Brian Caffo's Mathematical Biostatistics Boot Camp



* [Johns Hopkins](https://www.jhsph.edu/departments/biostatistics/index.html)
* [Coursera](https://www.coursera.org/learn/biostatistics/)
* [Github](https://github.com/bcaffo/Caffo-Coursera/)
* [Youtube](https://www.youtube.com/user/bcaffo/featured)

## Introduction


Things to keep in mind during the class:


* What is being modeled as random?

* From where does the randomness arise?

* From where does the systematic model components arise?

* [...]



### Experiments


For a given experiments:

* attribute all that is known or theorized to the systematic model

* attribute everything else to randomness

* use probability to quantify uncertainity in your conclusions

* evaluate the sensitivity of your conclusions to the assumptions of your model


### Set notation


### Probability

A **probability measure** is a function $P: \Omega \mapsto \mathbb{R}_{[0,1]}$, so that:

1. For an event $E \subset \Omega$, $0 \leq P(E) \leq 1$

2. $P(\Omega) = 1$

3. If $E_1$ and $E_2$ are mutually exclusive events, then $P(E_1\cup E_2) = P(E_1) + P(E_2)$


As a consequence (prove it!),


* $ P \big( \{\varnothing\} \big) = 0 $

* $ P(E) = 1 - P(\overline{E}) $

* $ P(A \cup B) = P(A) + P(B) - P(A \cap B)$

* if $ A \subset B $ then $ P(A) \leq P(B) $

* $ P(A \cup B) = 1 - P(\overline{A} \cap \overline{B}) $

* $ P(A \cap \overline{B}) = P(A) - P(A \cap B) $

* $ P(\bigcup_{i=1}^{n} E_i) \leq \sum_{i=1}^{n} P(E_i) $

* $ \max_{i} P(E_i) \leq \sum_{i=1}^{n} P(E_i) $


### Random variables

A **random variable** is an outcome of an experiment –– it can be *discrete* or *continuous*.



### PMFs and PDFs

A **Probability Mass Function** evaluated at a *value*, corresponds to the *probability* that a *random variable* takes that *value*.


To be a valid *pmf* a function, $p$, must satisfy


1. $$ p(x) \geq 0, \ \ \forall \ \ x $$

2. $$ \sum_{x} p(x) = 1 $$ sum is taken over **all possible values** of $x$


The canonical example is the Bernouili trial; let $X$ be the result of a coin flip where $X=0$ represents *Tail* and $X=1$ represents *Head*. Let $\theta$ be the probability of a head, then a valid *pmf* is

$$ p: x \mapsto \theta ^ {x} \ (1 - \theta ^ {1-x}) $$


We then have, probability of head is $p(1) = \theta$, and probability of tail is $p(0) = 1 - \theta$.


A **Probability Density Function** is a function associated with a *continous random variable*. Probability for the *random variable* corresponds to area under *pdf* curve. To be a valid *pdf* a function, f, must satisfy:


1. $$ f(x) \geq 0, \ \forall \ x $$

2. $$ \int_{-\infty}^{\infty} f(x)dx = 1 $$


### CDFs, survival functions and quantiles
