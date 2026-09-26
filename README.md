# BB gym - a gym / gymnasium environment for Bicycle Bivariate codes to be learned by Reinforcement Learning.

## What's in here ?
This is a Gymnasium (replacement for OpenAI's Gym) compliant environment that allows learning Bicycle Bivariate codes.


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

