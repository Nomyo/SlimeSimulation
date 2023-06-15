# SlimeSimulation
In my journey to learn more about Vulkan compute capabilty I step upon the famous slime simulation based on https://cargocollective.com/sagejenson/physarum and decided to give it a try with graphics and compute shaders.

## Samples

For the below example I have set all the particles (about 1 milion) uniformly distributed in a cercle centered in the frame. After a couple of secondes we can see an interesting pattern getting formed.
I have set a UI to modify multiple parameters in order to play a bit with the simulation. The different parameters are:
1. Speed of particles
2. Angular speed of particles
3. Diffuse rate that defines how much diffused the trail of each particle is diffused 
4. Decay rate, how long a trail of a particle remains before being dissipated
5. Sensor modifier: angle offset, distance to the particle and their size


<p align="center">                                                                                                                                                      
<img src =samples/SlimeSimulation_init_state.png/>                                                    
</p>

The particles moves according to their parameters and the rules of the simulation leading to a break in the equilibrium and the particles start to spread and explore further the given area.
To make it fun to watch I have set a particle to be a mutation in green that is 1.25x faster than the other particles. When the mutated particles enters in contact to another sain one it has a change to mutates it.

<p align="center">                                                                                                                                                      
<img src =samples/SlimeSimulation_2.png/>                                                                                                      
</p>

All particles are mutated and it is beautiful :).

<p align="center">                                                                                                                                                      
<img src =samples/SlimeSimulation_infected.png/>                                                    
</p>
