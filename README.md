## Sources


[Probability rule](http://ais.informatik.uni-freiburg.de/teaching/ss10/robotics/etc/probability-rules.pdf)


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


$$ E \Big[ \ \frac {1} {n} \sum_{i=1}^{n} X_i  \ \Big] = \frac {1} {n} \sum_{i=1}^{n} E[X_i] = \mu $$


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

For $k=3$, *Chebyshev* says the probability for a value to lie within $3$ standard deviations is $88.89\%$, whereas we know that probability for a normal distribution is $99.73\%$ –– using the [$68/95/99.7$ rule](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule).


## Random vectors

**Random vectors** are simply *random variables* collected into a vector. E.g. if $X$ and $Y$ are random variable then the vector $[X, Y]$ is a *random vector*.


For **continuous random vector** the joint density satisfies,

$$ f(x, y) \geq 0 \ \textrm{and} \iint f(x,y)dxdy = 1 $$


For **discrete random variable**,

$$ \sum \sum f(x,y) = 1 $$


## Independence

Two events $A$ and $B$ are **independent** if,

$$ P(A \cap B) = P(A)P(B)$$


Two random variables, $X$ and $Y$ are independent if for any two sets $A$ and $B$,

$$ P([X \in A]\cap [Y \in B]) = P(X \in A)P(Y \in B) $$

For a random vector, independence assumption is,

$$f(x,y)=f(x)f(y)$$


If $A$ and $B$ are independent then any of their subsets are also independent.


If a collection of random variables $\{X_i\}_{i=1\dots n}$ are independent, then their joint distribution is the product of their individual densities or mass functions or mass functions, $f_i$,

$$ f(x_1,\dots, x_n) = \prod_{i=1}^{n}f_i(x_i)$$


Moreover,
* if $f_1=\dots=f_n$, we say that $\{X_i\}_{i=1\dots n}$ are $iid$ ––**independent and identically distributed**,
* $iid$ random variables are the default model for random sample,



## Covariance

The **covariance** between two random variables  $X$ and $Y$ is defined as,

$$ Cov(X,Y) = E[(X-\mu_x)(Y-\mu_y)]$$


also equivalent to,

$$ Cov(X,Y)=E[XY] - E[X]E[Y] $$





Useful facts about covariance,

* $Cov(X,Y)=Cov(Y,X)$

* $|Cov(X,Y)| \leq \sqrt{Var(X)Var(Y)}$



## Correlation

The Correlation between $X$ and $Y$ is,

$$ Cor(X,Y) = \frac{Cov(X,Y)}  {\Big [ \ Var(X)Var(Y) \ \Big]^2}$$


Useful facts,


* $|Cor| \leq 1$

* $Cor(X,Y) = \pm 1  \iff \exists \ (a,b) \in \mathbb{R^2} \ \big | \ Y = aX + b$

* $Cor(X,Y) = 0 \implies X,Y$ are uncorrelated

* $Cor \rightarrow 1 \ (resp. -1)$: postive (*resp.* negative) correlation


## Variance and correlation properties


* when $\big \{ X_i \big \}$ are uncorrelated $Var(\sum_{i=1}^{n}a_i X_i +b) = \sum_{i=1}^{n}a_i^2Var(X_i)$

* ... otherwise add the term $2\sum_{i=1}^{n-1} \sum_{j=i}^{n}a_i a_j Cov(X_i, X_j)$
2

* **[Important]** if the ${X_i}$ are iid with variance $\sigma^2$ then $Var(\overline{X})=\sigma^2/n$ and $E[S^2]=\sigma^2$




## Variances properties of sample means

Suppose $X_i$ are iid with variance $\sigma ^ 2$, then,



$$Var(\overline{X}) = \frac {1}{n^2} \sum_{i=1}^{n} Var(X_i) = \frac {\sigma ^ 2} {n} $$




* when $X_i$ are independent with a common variance $Var(\overline{X}) = \sigma ^ 2 / n$
* $\sigma / \sqrt n$ is called **the standard error** of the sample mean,
* the standard error of the sample mean is the standard deviation of the distribution of the sample mean,
* $\sigma$ is the standard deviation of the distribution of a single observation,
* the sample mean has to be less variable than a single observation, therefore its standard deviation is divided by a $\sqrt n$

The variance of the sample means is also called Sampling Variance.

Sampling variance refers to variation of a particular statistic (e.g. mean) calculated in sample if you repeat the study many times.

**Sampling variance** is NOT  **Sample Variance**

## The sample variance

The **Sample Variance** is defined as


$$ S^2 = \frac {\sum_{i=1}^{n} (X_i - \overline{X})^2} {n-1} $$

This is a definition, just accept it! The sample variance is the variation in a single sample. If the population variance is not available this is what you use to estimate the standard deviation of the distribution of a single observation.



## Conditional Probabilities
[sce](https://github.com/bcaffo/Caffo-Coursera/blob/master/lecture5.pdf)

The formula to remember is,

$$ P(A \cap B) = P(A) P(B|A) $$


Therefore,

$$ P(B|A) = \frac {P(A \cap B)} {P(A)} $$


And if $A$ and $B$ are independent then,

$$ P(B|A) = P(B) $$


## Conditional densities

(a) Let $f(x,y)$ the bi-variate density of two RVs $X$ and $Y$,
(b) let $f(x)$ and $f(y)$ be the associated marginal mass function:

$$f(y) = \int{f(x,y)dx}$$

or


$$f(y) = \sum_{x}f(x,y)$$

Then the **conditional density** function at the point $Y=y$ is given by,

$$f(x|Y=y) = \frac {f(x,y)} {f(y)} $$


## Baye's rule

(a) let $f(x|Y=y)$ be the conditional density (*resp.* mass) function for $X$ given $Y=y$,
(b) let $f(x)$ be the marginal distribution for $x$

Then in the continous case,

$$ f(x|Y=y) = \frac {f(y|x)f(x)} {\int_t{f(y|t)f(t)dt}} $$


and in the discrete case,

$$ f(x|Y=y) = \frac {f(y|x)f(x)} {\sum_{t}{f(y|t)f(t)}} $$


Using the discrete version of the formula, it comes the,

$$ P(B|A)=\frac {P(A|B)P(B)} {P(A|B) P(B) + P(A|\overline{B})P(\overline{B})} $$

@TODO: the diagnostic tests from Lecture #5


## Likelihood
[sce](https://github.com/bcaffo/Caffo-Coursera/blob/master/lecture6.pdf)


We assume data comes from a family of distributions indexed by a parameter that represents a useful summary of the distribution. Therefore,


> The **Likelihood** of a collection of data is the joint density evaluated as a function of the parameters with the data fixed.


Another way to say it,

> Given a stistical probability mass function or density: $f(x, \theta)$, where $\theta$ is an unknown parameter, the **likelihood** is $f$ viewed as a function of $\theta$ for a fixed, observed value of $x$.


@TODO: confirm the following: frequentists assume the likelihood to contain to contain all relevant information regarding the model that generated the data:
The Likelihood Principle



## Some common distributions
[sce](https://github.com/bcaffo/Caffo-Coursera/blob/master/lecture7.pdf)


### Bernouilli distribution

The **Bernouilli distribution** arises as the result of a binary outcome. *Bernouilli* random variables take only the values $1$ and $0$ with a probability of $\theta$ and $1-\theta$ respectively.

The *PMF* for a *Bernouilli* random variable $X$ is

$$ P(X = x) = \theta^x (1 - \theta)^{1-x}$$


The *mean* of a *Bernouilli* random variable is $\theta$ and the *variance* is $\theta(1 - \theta)$

#### $iid$ Bernouilli trials


If several $iid$ *Bernouilli* observations, say $\{x_i\}_{i=1\dots n}$, are observed, the likelihood is:


$$ \prod_{i=1}^{n} \theta^{x_i}(1-\theta)^{1-x_i} = \theta^{\sum x_i} (1 - \theta)^{n -\sum{x_i}}$$

Notice from previous equation, that the likelihood depends only on $\sum x_i$. Because $n$ is fixed and assumed known, this implies that the sample proportion $\sum_i x_i /n$ contains all the relevant information about $\theta$.

If we maximize the Bernouilli likelihood over $\theta$, we obtain that $ \hat{\theta} = \sum_i x_i / n$ is the **maximum likelihood** estimator for $\theta$.


#### Binomial trials

The **Binomial random variables** are obtained as the sum of $iid$ *Bernouilli* trials. The *binomial* mass function is

$$ P(X=x) = \binom {n}{x} \ \theta^x (1 - \theta)^{n-x} $$, for $x=0\dots n$


#### Note for myself

**$iid$ Bernouilli** is different from **Binomial** counterpart, because the latter has the notion of *sum*, while the former deals with **one** observation...






### The Normal distribution

A random variable is said to follow a **Normal** or **Gaussian** distribution with mean $\mu$ and variance $\sigma^2$ if the associated density is,

$$ f(x) = \frac {1} {\sqrt{2 \pi }\sigma } e^{-\frac {(x - \mu) ^ 2} {2 \sigma ^ 2}} $$


If $X$ is a random variable with this density, we write $X \sim \mathcal{N} (\mu, \sigma)$, then $E[X]=\mu$ and $Var(X)=\sigma^2$.


When $\mu=0$ and $\sigma=1$ the resulting distribution is called the **Standard Normal distribution**, it is labelled $\phi$; standard normal variables are often labelled $Z$.


Useful facts,

* if $X \sim \mathcal{N} (\mu, \sigma ^2)$, then $Z=\frac {X-\mu}{\sigma}$ is standard normal,
* if $Z$ is standard normal then $X=\mu + \sigma Z \sim \mathcal{N}(\mu, \sigma^2)$
* **NO COMPRENDO** the non-standard normal density is $\phi \big \{   (x-\mu) / \sigma   \big \}   / \sigma$



### The Poisson distribution

// from Cameron's Bayesian for hackers


$Z$ is Poisson distributed if,


$$ P(Z=k) = \frac {\lambda^{k} e^{-\lambda}} {k!} $$

also noted as,



$$ Z \sim Poi(\lambda) $$



where,

* $k$ is a positve integer.

* $\lambda$ is called the parameter of the distribution, it controls the distribution's shape. $\lambda$ can be any positive number. By increasing $\lambda$ we add more probability to larger values; by decreasing it we add more probability to smaller values. $lambda$ is the intensity of the Poisson distribution.



A useful property of the Poisson distribution is that its expected value is equal to its parameter,

$$ E[Z | \lambda] = \lambda $$





## The Law of Large Numbers

> If $\big \{ X_i \big\}$ are iid from a population with mean $\mu$ and variance $\sigma ^ 2$ then $\overline{X}_n$ converges to $\mu$.




## The Central Limit Theorem

> The distribution of averages of iid variables, properly normalized, becomes that of a standard normal as the sample size increases.


## Confidence intervals using CLT

According to the CLT, the probability that the random interval


$$\overline{X}_n \pm z_{1-\alpha / 2} \ \sigma / \sqrt n $$


contains $\mu$ is approximately $95\%$, where $z_{1-\alpha / 2}$ is the $1-\alpha / 2$ quantile of the standard normal distribution.



## Sample proportions

In case of Bernouilli RVs with probability $p$,  $\sigma ^ 2 = p(1-p)$. The confidence interval is,

$$ \hat{p} \pm z_{1-\alpha / 2} \sqrt{\frac {p(1-p)} {n}}$$



## Confidence Interval take-2


We already see how to build CI using CLT, now let's see how it works for small sample size.


Gosset's **t-distribution** allows for better confidence intervals for small samples. In order to introduce **t-distribution** we must discuss the **Chi-Squared** distribution.



The general procedure for creating CIs:


* create a pivot statistic that does not depend on the parameter of interest,

* **NO COMPRENDO** solve the probability that the pivot lies between bounds for the parameter.


### The Chi-Squared distribution

Let $S^2$ be the sample variance from a collection of $iid$,  $\mathcal{N} (\mu, \sigma)$ data, then

$$\frac {(n-1) \ S^2} {\sigma ^ 2} \sim \chi^2_{n-1} $$

reads: follows a Chi-squared distribution with $n-1$ dof



Note:

* the **Chi-squared** distribution is skewed and has support on $0$ to $\infty$
* its mean is the dof : $n-1$
* its variance is twice the dof : $2 \times (n-1)$


Important note:

Let $\chi^{2}_{n-1, \alpha}$ be the $\alpha$ quantile of the Chi-squared distribution then,


$$1-\alpha = P\Big( \chi^{2}_{n-1, \ \alpha/2} \leq \frac {(n-1) \ S^2}{\sigma ^2} \leq \chi^{2}_{n-1, 1-\alpha/2}     \Big)$$


$$\ \ \ \  = P \Big( \frac{(n-1) \ S^2}{\chi^{2}_{n-1, \ 1-\alpha / 2}} \leq \sigma^2 \leq  \frac{(n-1) \ S^2}{\chi^{2}_{n-1, \ \alpha / 2}}   \Big)$$


so that

$$\Big [ \frac {(n-1) \ S^2} {\chi^2_{n-1, \ 1-\alpha/2}} \ , \  \frac {(n-1) \ S^2} {\chi^2_{n-1, \ \alpha/2}}  \Big]$$



is a $100 \ (1-\alpha)\ \%$ confidence interval for $\sigma ^ 2$



* this interval relies heavily on the **assumption of normality**,
* it turns out,


$$(n-1) \ S^2 \sim Gamma\{ (n-1) \ / \ 2, \ 2\sigma ^ 2 \}$$

which reads: follows a gamma distribution with shape $(n-1)/2$ and scale $2\sigma^2$

* the previous result can be used to plot a likelihood function for $\sigma ^2$


### Gosset's $t$ distribution

Invented by William Gosset in 1908, it has thicker tails than the normal distribution. It is indexed by a degrees of freedom; gets more like a standard normal as df gets larger. The formula,


$$ \frac {Z} {\sqrt{ \chi^2 \ / \ df}} $$


where $Z$ and $\chi^2$ are independent standard normals and Chi-squared distributions respectively.




Let $(X_1, \dots, X_n)$ iid and $\mathcal{N}(\mu, \sigma^2)$, then:

1. $\frac{\overline{X}-\mu}{\sigma / \sqrt{n}}$ is standard normal,
2. $\sqrt{\frac{(n-1) \ S^2}{\sigma ^2 (n-1)}} = S \ / \ \sigma$ is the square-root of a Chi-squared divided by its dof.



Then,

$$\frac { \frac {\overline{X}-\mu} {\sigma / \sqrt{n}} } {S/\sigma} = \frac {\overline{X}-\mu} {S/\sqrt{n}} $$


follows Gosset's $t$ distribution with $n-1$ dof.


We can use it to create a confidence interval for $\mu$.


Let $t_{df, \alpha}$, bt the $\alpha^{th}$ quantile of the $t$-distribution with given dof,

$$ 1-\alpha = P\Big(  -t_{n-1, 1-\alpha/2} \leq \frac {\overline{X} - \mu} {S/\sqrt{n}}  \leq t_{n-1, 1-\alpha/2}  \Big)$$

$$ \ \ = P \Big(  \overline{X} - t_{n-1, 1-\alpha/2} \ S \ / \ \sqrt{n} \leq \mu \leq \overline{X} + t_{n-1, 1-\alpha/2} \ S \ / \sqrt{n} \Big)$$



The interval is $\overline{X} \pm t_{n-1, \ 1-\alpha/2} \ S / \sqrt{n}$




Important note:



* the $t$ interval technically assumes that the data are iid normal,
* though it is robust to this assumption, it works well whenever the distribution of the data is roughly symmetric and mound shape
* Paired observations are often analyzed using the $t$ interval by taking differences
* For large degrees of freedom, t quantiles become the same as standard normal quantiles; therefore this interval converges to the same interval as the CLT yielded
* For skewed distributions, the spirit of the $t$ interval assumptions are violated
* Also, for skewed distributions, it doesn't make a lot of sense to center the interval at the mean
* In this case, consider taking logs or using a different summary like the Median
* for highly discrete data, like binary, other intervals are available



@TODO: finish lecture 9: Profile likelihood.


## TODO: Lecture 10 - T-Confidence Interval

## TODO: Lecture 11 - Plotting



## The Jackknife

The **Jackknife** is a small tool for estimating standard errors and the bias of estimators. It involves *resampling* data.


How it works?

* The jackknife deletes each observation and calculates an estimate based on the remaining $n-1$ of them,
* this collection of estimates is used to estimate things like **bias** and **standard error**,
* Note that bias and standard error are not needed for sample means (because it is unbiased estimates of pupulation means and their standard errors are known),


an example?

* let's consider univariate data,
* let $X_i$ be a collection of data used to estimate a parameter $\theta$,
* let $\hat{\theta}$ be the estimate based on the full data set,
* let $\hat{\theta}_i$ be the estimate of $\theta$ obtained by deleting observation $i$,
* let $\overline{\theta} = \frac{1}{n} \sum_{i=1}^{n} \hat{\theta}_i$
* then, the jackknife estimate of the bias is

$$ (n-1) \big ( \ \overline{\theta} - \hat{\theta} \ \big ) $$



* The jackknife estimate of the standard error is

$$ \big [  \frac {n-1} {n} \sum_{i=1}^{n} (\hat{\theta}_i - \overline{\theta})^2 \big ] ^ {1/2} $$


## The bootstrap

* the bootstrap principle suggests using the distribution defined by the data to approximate its sampling distribution.

* In practice bootstrap is always carried using simulation,

* the general procedure follows by first simulating complete data sets from the observed data with replacement,

* calculate the statistic for each simulated data set,

* use the simulated statistics to either define a confidence interval or take the standard deviation to calculate a standard error.




## Intervals for Binomial proportions

**{Important}**

* When $X \sim Binomial(n, p)$ we know that,
  a. $\hat{p} = X/n$ is the MLE for p,
  b. $E[\hat{p}] = p$
  c. $Var(\hat{p}) = p(1-p)/n$
  d. $\frac {\hat{p} - p} {\sqrt{\hat{p}(1-\hat{p})/n}}$


* the latter fact leads to the Wald interval for $p$

$$ \hat{p} = Z_{1-\alpha / 2}  \sqrt{\hat{p}(1-\hat{p}) / n} $$


@TODO read again about Wald interval and how it is fixed with Agresti-Coull intervals.



## Bayesian analysis

some more vocabulary

* Bayesian statistics posits a **prior** on the parameter of interest,
* all inferences are then performed on the distribution of the parameter given the data, called the **posterior**
* In general,


$$ {Posterior} \propto {Likelihood} \times {Prior} $$


* the likelihood is the factor by which our prior beliefs are updated to produce conclusions in the light of the data.


## Beta priors

* It is the default **prior** for parameters between $0$ and $1$.
* the Beta density depends on 2 parameters $\alpha$ and $\beta$,


$$  Beta(\alpha, \beta) = \frac {\Gamma(\alpha + \beta)} {\Gamma(\alpha) \Gamma(\beta)} p^{\alpha -1}(1-p)^{\beta - 1} \ , \ \ 0 \leq p \leq 1 $$




* the mean of the Beta density is


$$\alpha / (\alpha + \beta)$$


* the variance of the Beta density is


$$ \frac { \alpha \beta } {(\alpha + \beta)^2 (\alpha + \beta + 1)} $$



* the uniform density is the special case where $\alpha=\beta=1$


## Beta posterior

The resulting **posterior** is also a Beta function,

$$Posterior \propto p^{x+\alpha-1}(1-p)^{n-x+\beta-1}$$


## Bayesian credible intervals

* a Bayesian credible interval is the Bayesian analog of a confidence interval,
* a $95\%$ credible interval, $[a,b]$ would satisfy


$$ P(p \in [a,b] \ | x) = 0.95 $$


* the best credible intervals chop off the posterior with a horizontal @TODO: why?







## Hypothesis testing

Condider court of law

* the **null hypothesis** is the defendant is innocent,
* evidence is required to reject the **null hypothesis**,
* **type I error** = percentage of innocent people convicted
* **type II error** = percentage of guilty people let free
* too little evidence required increases **type I error**,
* too much evidence required increases **type II error**



Example,


* A respiratory disturbance index of more than 30 events / hour is considered evidence of severe sleep disordered breathing.

* In a sample of 100 overweight subjects with other risk factors the mean RDI was 32 events / hour with a standard deviation of 10 events / hour.

* Our hypothesis ($\mu$ is the population mean RDI)
  * $H_0: \ \mu=30$
  * $H_a: \ \mu > 30$


Solution,

* a reasonable strategy is would be reject the null hypotheisis if $\overline{X}$ was larger than some constant $C$

* $C$ is chosen such that the probability of a **Type I error**, $\alpha$ is $0.05$,

$0.05 = P \big (\overline{X} \geq C | \mu = 30  \big)$
$     = P\Big ( \frac {\overline{X} - 30} {10/\sqrt{100}} \geq \frac {C-30} {10/\sqrt{100}} | \mu =30 \Big )$
$     = P\big ( Z \geq \frac {C-30} {1} \big ) $
$     = 0.05 = P(Z \geq 1.645)$

Hence $C=31.645$, we reject the null hypothesis $H_0$



Note it is enough to remark that


$$\frac {32-30} {10/\sqrt{100}}=2 \geq 1.645$$



Ce qu'il faut retenir,


* the $Z$ test for $H_0: \mu=\mu_0$
  * $H_1: \mu < \mu_0$
  * $H_2: \mu \neq \mu_0$
  * $H_3: \mu > \mu_0$

* the test statistic: $TS = \frac {\overline{X} - \mu_0} {S/\sqrt{n}}$

* reject the null hypothesis when,
  * $H_1: TS \leq -Z_{1-\alpha}$
  * $H_2: |TS| \geq Z_{1-\alpha / 2}$
  * $H_3: TS \geq Z_{1-\alpha}$




**[IMPORTANT]**

* the Z-test requires the assumption of the CLT and for n large enough for it to apply,

* if n is small, then Gossett's T test is performed exactly the same way, with the normal quantiles replaced by the appropriate Student's T quantiles and n-1 dof


**[IMPORTANT]**
* **Power** is the probability of rejecting the null hypothesis when it is false
* **Power** is used a lot to calculate sample size for experiments



Example reconsidered

Let's suppose n=16 (rather than 100), then,

$$ 0.05 = P \big (  \frac {\overline{X} - 30} {s/\sqrt{16}} \geq t_{1-\alpha, 15} | \mu =30 \big ) $$

$\sqrt {16} (32-30)/10 = 0.8$ and $t_{1-\alpha, 15}=1.75$, we fail to reject the null hypothesis.



## Hypothesis Testing: CI and P-values

**Confidence Intervals**
The set of all possible values for which you fail to reject the $H_0$ is a $(1-\alpha)\times 100\%$ **Confidence Interval** for $\mu$,



Similarly, if a $(1-\alpha) \times 100\%$ interval contains $\mu_0$ then we **fail to** reject $H_0$,



**P-values**

The **P-value** is the probability under the null hypothesis of obtaining evidence as extreme or more extreme than would be observed by chance alone.




Notes,


* report the P-value, you can perform the hypothesis test at whatever $\alpha$ level you want,

* if the P-value is less than $\alpha$ you reject the null hypothesis,

* for two sided hypothesis test, double the smaller of the two onde-sided hypothesis P-value,

* **Bottom line is** do not just report P-values, give CIs too!





## Power
[sce](https://github.com/bcaffo/MathematicsBiostatisticsBootCamp2/blob/master/lecture2.pdf)

$$Power = 1-\beta$$

* **Power** is the porbability to reject the null hypothesis when it is false,
* **Power** is good, you want more of it,
* A type-II error is failing to reject the null hypothesis when it is actually false: it is noted $\beta$


## T-test



## Monte Carlo



## Two sample Tests



























# More wordings

http://blog.minitab.com/blog/adventures-in-statistics-2/understanding-hypothesis-tests-confidence-intervals-and-confidence-levels

## Significance level


The significance level, also denoted as $alpha$, is the probability of rejecting the null hypothesis when it is true. For example, a significance level of 0.05 indicates a 5% risk of concluding that a difference exists when there is no actual difference.

significance level <--> type I error




## P-values

P-values are the probability of obtaining an effect at least as extreme as the one in your sample data, assuming the truth of the null hypothesis.

When a P value is less than or equal to the significance level, you reject the null hypothesis.


## Statistical significance

A hypothesis test evaluates two mutually exclusive statements about a population to determine which statement is best supported by the sample data. A test result is statistically significant when the sample statistic is unusual enough relative to the null hypothesis that we can reject the null hypothesis for the entire population. “Unusual enough” in a hypothesis test is defined by:

* The assumption that the null hypothesis is true—the graphs are centered on the null hypothesis value.
* The significance level—how far out do we draw the line for the critical region?
* Our sample statistic—does it fall in the critical region?


## Confidence intervals


## Confidence level
