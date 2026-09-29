# BB gym - a gym / gymnasium environment for Bicycle Bivariate codes to be learned by Reinforcement Learning.

## What's in here ?
This is a Gymnasium (replacement for OpenAI's Gym) compliant environment that allows learning Bicycle Bivariate codes.

This is the code for the environment used in the paper [Designing Quantum Error Correcting Codes to fit decoders via Reinforcement Learning](https://arxiv.org/abs/2608.15754)

Note that at the moment the decoder used for evaluation is hardwired to be a BP+OSD decoder, but the option to plug an arbitrary decoder will be available soon (as long as the API for the decoder is met).

This means that we define:
1. Actions
2. Observations
3. States
4. Reward
5. Truncation (in this environment there is no termination, which is slightly different).

And provide the functions:
1. step - takes an action and applies it to the environemnt. Returns observation, reward, and whether or not truncation or termination is flagged.
2. reset - resets the environment to an initial value.



## What are BB codes ?
The usual reference for BB codes for the quantum setting is 
[High-threshold and low-overhead fault-tolerant quantum memory](https://arxiv.org/pdf/2308.07915)
By Sergey Bravyi, Andrew W. Cross, Jay M. Gambetta, Dmitri Maslov, Patrick Rall, and Theodore J. Yoder

In this environment, we generalised the construction slightly to:

Start with prescribed (binary) matrices $𝑥,𝑦$ such that $$𝑥^𝑙=y^𝑚=𝐼_{𝑙𝑚}$$ 
for some $𝑙,𝑚$ and such that $$𝑥𝑦=𝑦𝑥$$

Define four binary polynomials: 
$$a(X)=∑𝑎_𝑖 X^𝑖, \qquad deg⁡(𝑎(X))≤𝑙 $$

$$a(Y)=∑a'_𝑖 Y^𝑖, \qquad deg⁡(a(Y))≤𝑚 $$

$$b(X)=∑b_𝑖 X^𝑖 ,\qquad deg⁡(b(X))≤𝑙 $$

$$b(Y)=∑b'_𝑖 Y^𝑖,\qquad deg⁡(b(Y))≤𝑚 $$

Take: 

$$𝐴=𝑎(𝑥)+a(𝑦)$$

$$𝐵=b(𝑥)+b(𝑦)$$

And finally: 

$$𝐻^𝑋=[𝐴│𝐵]$$

$$𝐻^𝑍=[𝐵^𝑇|𝐴^𝑇]$$

Then 
$$𝐻^𝑋 \cdot (𝐻^𝑍)^𝑇=𝐴\cdot𝐵^𝑇−𝐵^𝑇\cdot 𝐴=0$$

And so they define a CSS code.

Also note some properties:

$$𝑟𝑎𝑛𝑘(𝐻^𝑋 )=𝑟𝑎𝑛𝑘(𝐻^𝑍)$$

And therfore, the code, which physical dimension (number of physical qubits) is 

$$N = 2lm$$

And which (always) has

$$K = N-rank(H^X)-rank(H^Z)$$

logical qubits, satisfies:

$$K = 2lm - 2*rank(H^X) $$

# So what is an action in this context ?
An action $A^p$ on a binary polynomial $p(t)$ flips at most one of its coefficients.

We can subsequently define an action on the polynomials in the above definition to be a tuple of individual actions $A^{a(x)},A^{b(x)},A^{a(y)},A^{b(y)}$

# What are the states and the observations ?
The state follows from the polynomials, and is the matrices $H^X, H^Z$.

The observation includes the matrices $A,B$, as well as the new polynomials.

# How is the reward defined, and how is it calculated ?
In [Designing Quantum Error Correcting Codes to fit decoders via Reinforcement Learning](https://arxiv.org/abs/2608.15754) we define the reward as a function of decoder evaluation, i.e.:

1. We sample errors from a region of intereset of physical error rates (using the depolarizing channel, although this could be extended).
2. We try to decode, and see if the decoder produced a correction error which, in conjuction with the sampled error, produces a stabilizer.
3. We do this repeatedly, and get a physical to logical error curve.
4. We integrate to get the area above the curve, and divide by the width of the physical error rate region (constant through the life cycle of the environment). This means that a higher reward represents less logical errors in the region of interest, for a fixed decoder and the present state code.