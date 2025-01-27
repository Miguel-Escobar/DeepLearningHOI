# Higher Order Interactions in Deep Continual Learning

This repo centains code for a reserch project started at the LACONEU school of Computational Neuroscience. It **heavily** utilizes code and ideas from the the paper [Loss of Plasticity in Deep Continual Learning](https://doi.org/10.1038/s41586-024-07711-7) and it's associated [github repo](https://github.com/shibhansh/loss-of-plasticity/tree/main).

## A brief goal statement of the project:

The aim is to investigate whether higher order information metrics, such as O-info, TC, DTC, etc..., give us some insight into what happens to a continously trained network that makes it be less plastic over training time. Based on what we see we can choose whether to leave it at that or go further.

## Slowly Changing Regression:

The task to study is the _slowly changing regression_ mentioned in the original article. It consists in fitting a particularly small, 5 hidden unit, feed forward neural network (hereafter referred to as student) to a target 100 hidden unit, single layer neural network (whose parameters are fixed during training, and who will be hereafter referred to as teacher). This ensures that the teacher function can't be completely fit by the student function. 

We then train the student by taking samples from particular subsets of the possible inputs, obtained by fixing the last 15 bits of the input, leaving 5 bits free to quickly sample the space. After a large number of steps (10000 for instance) we change one of the 15 fixed bits. 

This training method, coupled with the fact that the student will never fit the whole teacher, ensures that we continously train the student, emulating with a computationally cheap task the phenomenon of continual learning and loss of plasticity.

## Higher Order Information Metrics:

**Pendiente**
