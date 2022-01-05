<h2>cDFT</h2>
<h4>Mary K. Coe

December 2021</h4>

cDFT is a Python package to implement classical Density Functional Theory (cDFT) for fluids 
in contact with smooth planar surfaces, smooth solutes and confined between two smooth planar
walls (an infinite slit).

I was introduced to cDFT during the early stages of my PhD. Whilst I could clearly see the 
merits of cDFT, I found that the literature surrounding it was fairly scattered and there
was limited information on how to actually implement it computationally. A lot of the numerical
tricks to write a good cDFT program are typically passed by word of mouth and I personally
have seen very few resources which clearly lay out the mathematics behind implementing 
attractive fluids. Obstacles such as these meant I found it difficult to firstly find a package
that I could use and then to write my own. I therefore decided that when I wrote my own program 
for cDFT I would try to make it as user-friendly and freely available as possible. 

As this package was written as part of my PhD research it is by no means professional or
to a large extent complete. In time I plan to add notes here with instructuctions on how to
add new fluid and external potentials, so that other students (or anyone else who is interested)
will be able to use this package in the way that best meets their research needs. 

In the meantime, this repository will consist of the package in its current state and three
tutorials which I hope will help you understand how to use it. For more information on the theory
behind this package, and details of underlying calculations, please see Chapter 4 and Appendices
A-C of my thesis, available at: https://research-information.bris.ac.uk/ws/portalfiles/portal/304220732/Thesis_Mary_Coe.pdf.


<h3>What systems can I investigate with this cDFT package?</h3>

This package is setup to deal with two fluid types and three geometries. The fluids that can be
studied using this package are the hard-sphere fluid and a truncated Lennard-Jones (LJ) fluid. Note
that the length of the truncation in the latter is arbitrary - you can set the truncation to be
hundreds of times the size of the fluid particle. This means the package can effectively study both
short-ranged and long-ranged LJ fluids.

The package gives the user the option of three systems: a fluid in contact with a smooth planar 
surface; a fluid in contact with a smooth solute of arbitrary radius; and a fluid confined to
an infinite slit in which the walls are smooth. The use of the term 'smooth' is important here.
This means that the external potential that any of these surfaces exerts on the fluid varies in
only one direction. For a planar surface or a slit, this is the direction perpendicular to the
wall(s). For a solute, this is along the radial axis extending from the centre of the solute.
The density profile of the fluid will take the symmetry of the external potential exerted on it,
hence for each of these systems the density profile varies along only one axis. The consequence
of this is that the weighted density and correlation functions can be reduced to one-dimensional
calculations by analytically evaluating their results in the two dimensions in which the density
profile is homogeneous. This greatly reduces the computation involved and therefore makes the
package fairly quick. To clarify, the systems studied are three-dimensional, we have just made the
computer's job easier by doing the calculation ourselves in two of the three dimensions.

<h3>How can I tell the numerical consistency of the calculations?</h3>

The package implements statistical mechanics sum rules. These are formally exact equations which
relate microscopic and macroscopic properties of a system. Importantly, either side of these 
equations can be calculated independently and hence you can test the numerical consistency of 
asking the package to calculating either side of these equations and compare them.

This package uses two different sum rules - the Gibbs' adsorption equation and the contact sum 
rule. For more information, take a look at the tutorials.


<h3>Will you be adding any more features to this package?</h3>

Honestly, probably not. This was done as part of my PhD which I have now finished, and I am likely
to move on to other challenges. However I encourage you to add your own features. In due course
I'll hopefully put some notes up to explain how to add features into the package.


<h3>How do I use the package and what features are currently available?</h3>

To find out how to use the package, take a look at the tutorials! I also provide references there
to help aid your understanding of cDFT.

The package currently supports:

<b>Fluids</b>: Hard-Sphere, truncated Lennard-Jones

<b>Geometries</b>: Planar surface, solute and infinite slit.

<b>Functionals (for implementing Fundamental Measure Theory for hard-spheres)</b>: Rosenfeld, White-Bear
  and White-Bear Mark II
  
<b>Surface/Solute-fluid interactions</b>: hard (planar, solute and slit), Lennard-Jones (planar and solute), 
  shifted Lennard-Jones (planar, solute and slit), WCA Lennard-Jones (planar)
  
<b>Sum-Rules</b>: contact sum rule and (Gibbs') adsorption sum-rule for all surface/solute-fluid interactions
  listed above
  
<b>Measures</b>: adsorption, surface tension, contact density, density profile, local compressibility and
  local thermal susceptibility profiles (see tutorial 3 for an explaination of these)
 
  






