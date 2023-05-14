# ProES - Prompt Emission Simulator
This code simulates the prompt emission phase of gamma-ray bursts. It was created as a part of my thesis for the Master of Physics (Astronomy & Astrophysics) at the University of Western Australia.

## About this code
ProES is a multi-purpose simulator of the prompt emission phase of gamma-ray bursts. It can generate a range of prompt light curves from varying GRB parameters, and can also accelerate protons within the jet. At this stage, the implemented proton energy loss mechanisms are synchrotron, dynamical, and photohadronic. 

In version 2.0.0 of this code to be released, I will allow for the user to generate detector-specific prompt light curves.

## Installation
To install, simply download the .zip of this repository and run the code. I have not yet generalised the import of the pysophia symbolic objects for the photohadronic loss timescales for all users, and this will be present in the next version. This code will eventually be installable via pip, and this feature will appear in version 1.0.0. 

## Documentation
The documentation is found in the ```simulation/docs/proes ``` folder in .html format. I am still in the process of finalising documentation of all functions, but what exists so far provides a general overview of how each class and function operates.

## Examples
To get started, look through and run the test_sim.py file located in ```simulation/proes```.

## Copyright and Licensing 
This code is released under the MIT license.

## Citation
Will publish a paper on this code at some point this year after my thesis is submitted :) 
