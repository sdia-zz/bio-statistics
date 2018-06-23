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



## Experiments


For a given experiments:

* attribute all that is known or theorized to the systematic model

* attribute everything else to randomness

* use probability to quantify uncertainity in your conclusions

* evaluate the sensitivity of your conclusions to the assumptions of your model


## Set notation


## Probability

A **probability measure** is a function $P: \Omega \mapsto \mathbb{R}_{[0,1]}$, so that:

1. For an event $E \subset \Omega$, $0 \leq P(E) \leq 1$

2. $P(\Omega) = 1$

3. If $E_1$ and $E_2$ are mutually exclusive events, then $P(E_1\cup E_2) = P(E_1) + P(E_2)$


As a consequence,


* $ P \big( \{\varnothing\} \big) = 0 $

* $ P(E) = 1 - P(\overline{E}) $

* $ P(A \cup B) = P(A) + P(B) - P(A \cap B)$

* if $ A \subset B $ then $ P(A) \leq P(B) $

* $ P(A \cup B) = 1 - P(\overline{A} \cap \overline{B}) $

* $ P(A \cap \overline{B}) = P(A) - P(A \cap B) $

* $ P(\bigcup_{i=1}^{n} E_i) \leq \sum_{i=1}^{n} P(E_i) $

* $ \max_{i} P(E_i) \leq \sum_{i=1}^{n} P(E_i) $


## Random variables

A **random variable** is an outcome of an experiment –– it can be *discrete* or *continuous*.



## PMFs and PDFs

A **Probability Mass Function** evaluated at a *value*, corresponds to the *probability* that a *random variable* takes that *value*.


To be a valid *pmf* a function, $p$, must satisfy


1. $$ p(x) \geq 0, \ \ \forall \ \ x $$

2. $$ \sum_{x} p(x) = 1 $$ sum is taken over **all possible values** of $x$


The canonical example is the Bernouili trial; let $X$ be the result of a coin flip where $X=0$ represents *Tail* and $X=1$ represents *Head*. Let $\theta$ be the probability of a head, then a valid *pmf* is

$$ p: x \mapsto \theta ^ {x} \ (1 - \theta ^ {1-x}) $$


We then have, probability of head is $p(1) = \theta$, and probability of tail is $p(0) = 1 - \theta$.


A **Probability Density Function** is a function associated with a *continous random variable*. Probability for the *random variable* corresponds to area under *pdf* curve. To be a valid *pdf* a function, f, must satisfy:


1. $$ f(x) \geq 0, \ \forall \ x $$

2. $$ \int_{-\infty}^{+\infty} f(x)dx = 1 $$


## CDFs and Survival functions


The **Cumulative Distribution Function** of a *random variable* $X$ is defined as

$$ F : x \mapsto P(X \leq x) $$

The **Survival Function** of a *random variable* $X$ is defined as
$$ S : x \mapsto P(X > x) $$

Remark:

* $S = 1 - F $

* for *continuous random variable* the *PDF* is the derivative of the *CDF*


## Quantiles

The $\alpha^{th}$ – ___quantile___ of a distribution with *CDF* $F$, is the point $x_{\alpha}$ so that


$$ F(x_{\alpha}) = \alpha $$


A **Percentile** is just a *quantile* expressed as percent – e.g. the **Median** is the $50^{th}$ – *percentile*.


Remark:

* the *median* discussed here is **population** quantity; it is different from the median you get from a sample data,

* a probability model connect data to the **population** using assumptions,

* the *population median* is reffered to as **estimand**, whereas the
sample median* will be the **estimator**.



## Expected values

The **Expected Value** ––also called *mean* –– of a *random variable* $X$, is the **barycenter** of its distribution.

For a **discrete** random variable

$$ E(X) = \sum_{x} x \ p(x)$$


For a **continuous** random variable

$$E(X) = \int_{-\infty}^{+\infty} t \ p(t) \ dt$$


Remarks:

The uniform function: $[0, 1] \mapsto 1$, (a) is a valid *density function* and (b) its *expected value* is $1/2$.

The *expected value* is a linear operator

* $E[X + Y] = E[X] + E[Y]$, where $X$ and $Y$ are random variables,

* $E[aX + b] = a E[X] + b$, where $a$ and $b$ are constant,


### Sample mean

Let us consider a **collection** of *random variable* $\{ X_i \}, \ i = 1\dots n$, with each having the same expected value $\mu$. Therefore the expected value of the sample average of the $\{ X_i \}$ is also $\mu$:


$$ E \big[ \frac {1} {n} \sum_{i=1}^{n} X_i  \big] = \frac {1} {n} \sum_{i=1}^{n} E[X_i] = \mu $$


Therefore the *expected value* of the **sample mean** is the population *mean* that it's trying to estimate. We then say: *the expected value is an **unbiased** estimator*.


### Application to Binomial

The *expected value* from the result of the toss of a coin with probability of heads (i.e. $X=1$) of $\theta$ is


$$E[x] = \sum_{i=0}^{1} i \ \theta ^ i \ (1 - \theta) ^ {1 - i} = 0 \ \times \ (1 - \theta) + 1 \times \theta = \theta $$


Therefore the expected value of a Binomial process is the probability of success.


## Variance

The **Variance** of a *random variable* is a measure of *spread*.
If $X$ is a random variable with mean $\mu$, the *variance* of $X$ is defined as


$$ Var : x \mapsto E\big[ \ (X - \mu) ^ 2 \ \big] $$


Remarks:



*Variance* is **NOT** a linear operator,  $Var[aX] = a^2 \ Var[X]$, where $a$ is constant


### Standard Deviation

Another way to say is the *variance* is the *expected* squared distance from the mean.

The square root of the variance has the same dimension as $X$ and is called the **Standard Deviation**:

$$ \sigma = \sqrt{Var} $$



### Computational form


A convenient computational form for the *variance* is

$$ Var : x \mapsto E[X^2] - \big( E[X] \big)^2 $$


E.g. the sample variance from the result of a toss of a die:


$$ E[X] = \sum_{i=1}^{6} \frac{1}{6} \ i = 3.5 $$

$$ E[X^2] = \sum_{i=1}^{6} \frac{1}{6} \ i^2 = 15.17 $$

and therefore,

$$ Var[X] = 15.17 - 3.5^2 = 2.92 $$


### Application to Bernouilli

The sample variance from the result of the toss of a coin with probability of heads (i.e. $X=1$) of $\theta$ is


$$E[x] = \theta $$


$$ E[X^2] = \sum_{i=0}^{1}  \theta ^ i \ (1 - \theta) ^ {1 - i} \ (i^2) = (1 - \theta) \times 0^2\ + \theta \times 1^2 = \theta $$


and therefore,


$$ Var[X] = \theta - \theta ^ 2 = \theta (1 - \theta) $$

The variance of a Bernouilli process with parameter $\theta$ is **$Var_{Ber} = \theta(1-\theta)$**

Remark:

Let us suppose a random variable $X$ such that $0\leq X \leq 1$ and $E[X] = \theta$, therefore:

$ E[X^2] \leq E[X] \Rightarrow E[X^2] \leq p $

$ Var[X] = E[X^2] - E[X]^2 \Rightarrow Var[X] \leq \theta - \theta^2 = Var_{Ber}$

In other words the *Bernouilli variance is the largest possible for random variables bounded between 0 and 1.*


## Chebyshev's inequality


Let $X$ be a *random variable* with finite *expected value* $\mu$ and finite non-zero *variance* $\sigma^2$. Then for any real number $k>0$,


$$P(|x - \mu| \geq k \sigma) \leq \frac {1} {k^2}$$


E.g. the probability that values lie outside the interval $\big( \mu - \sqrt{2} \sigma , \ \mu + \sqrt{2} \sigma \big)$ does not exceed $1/2$.


Because it can be applied to completely arbitrary distributions provided they have a known finite mean and variance, the inequality generally gives a poor bound compared to what might be deduced if more aspects are known about the distribution involved.

For $k=3$, the Chebyshev says the probabilty to lie within $3$ standard deviations is $88\%$, whereas we know for a normal distribution it is $99.73\%$ –– using the 68/95/99.7 rule.
