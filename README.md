# Higher Order Interactions in Deep Continual Learning

This repo contains code for a research project started at the LACONEU school of Computational Neuroscience. It **heavily** utilizes code and ideas from the the paper [Loss of Plasticity in Deep Continual Learning](https://doi.org/10.1038/s41586-024-07711-7) and it's associated [github repo](https://github.com/shibhansh/loss-of-plasticity/tree/main).

## A brief goal statement of the project:

The aim is to investigate whether higher order information metrics, such as O-info, TC, DTC, etc..., give us some insight into what happens to a continously trained network that makes it be less plastic over training time. Based on what we see we can choose whether to leave it at that or go further.

## Slowly Changing Regression:

The task to study is the _slowly changing regression_ mentioned in the original article. It consists in fitting a particularly small, 5 hidden unit, feed forward neural network (hereafter referred to as student) to a target 100 hidden unit, single layer neural network (whose parameters are fixed during training, and who will be hereafter referred to as teacher). This ensures that the teacher function can't be completely fit by the student function.

We then train the student by taking samples from particular subsets of the possible inputs, obtained by fixing the last 15 bits of the input, leaving 5 bits free to quickly sample the space. After a large number of steps (10000 for instance) we change one of the 15 fixed bits.

This training method, coupled with the fact that the student will never fit the whole teacher, ensures that we continously train the student, emulating with a computationally cheap task the phenomenon of continual learning and loss of plasticity.

## Higher Order Information Metrics:

In the study of complex systems, one is very frequently faced with convoluted relationships between groups of variables that are not simplifyable into pairwise interactions, at least not in a practical sense (as students, it would be interesting to know if there are interactions which are mathematically impossible to be represented via pairwise interactions).

We then are faced with the issue of measuring interactions between more than two variables, a task for which Shannon's information theory provides a helpful framework. Using this approach, one can define various metrics, of particular interest for this work is O-information.

### O-information:

If you consider that HOI are somewhere on a spectrum between having total shared randomness and having fixed collective constraints, one can devise the O-information to quantify where on this spectrum a system may lie. Now I will follow an article by Fernando Rosas et. al. to introduce this concept precisely:

Consider an n-dimensional random vector $X = (x_i)_{i = 1}^n$, each component taking values over a corresponding finite alphabet, $\mathcal{X}_i$. Assuming this is the only information available, interpreted as having a uniform prior distribution, the entropy of the random variable $x_i$ is simply $\log(|\mathcal{X}_i|)$. The _negentropy_ is defined as the difference between the uniform prior entropy and whatever entropy in calculated as with whatever other prior. It is literally the entropy but with a minus sign and a positive constant added to ensure positivity.

The difference between the whole system negentropy and the sum of the individual marginal negentropies gives rise to a measure of the strength of the _collective constraints_ of the system. This is known as the Total Correlation.

The dual quantity, obtained by measuring the marginal entropies, corresponds to data contained in a variable that cannot be extracted from measurements of other variables. The difference between the whole system entropy and the sum of these marginal entropies measures the amount of information contained in sets of two or more variables. This is known as the Dual Total Correlation, and can also be viewed as the _shared randomness_ contained in a system.

Now we can finally introduce the O-information, as the difference between the Total Correlation and the Dual Total Correlation. In short:

$$
\Omega(X^n) = TC(X^n) - DTC(X^n) = (n-2) H(X^n) + \sum_{j = 1}^n [H(X_j) - H(X^n_{-j})]
$$

Where $X^n_{-j}$ is the vector of $n-1$ variables that excludes $X_j$.
