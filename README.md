# basic_perceptron_network
Learning how to create a very basic perceptron network for machine learning

As an UG student I was never really given much opportunity to explore machine learning in much capacity, but always 
found it interesting. I have made it my new years resolution to pick up as much as I can in my own time before starting 
an MSc next year, so here we are at my humble beginning.

This is simply the start of picking up the basics of neural networks, starting with a *perception network*

### What is a perceptron network?

A perceptron network is a supervised learning algorithm. The algorithm is given inputs (*x*) and these inputs are  
multiplied by a weight (*w*). The sum of all the inputs multiplied by their weights is passed to the *perceptron
function*, which will calculate whether or not the neuron will activate (resulting in a boolean output).

A set of *training data* is given to the perceptron network, and starting weights are decided (in this case, randomly). 
The network will then compare its output with the expected output, and adjust the weights depending on how wrong the
output was. This process repeats until a satisfactory solution has been found.

![perceptron_network](perceptron_learning.jpg)
