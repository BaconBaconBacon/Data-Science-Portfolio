An example of a Python Jupyter notebook analysis from my MSc studies. It's currently in serious need of tidying and re-factoring.




# Introduction

The purpose of this script is to determine the symmetry of the Fermi surface of YbPdBi, a material which has demonstrated interesting quantum effects (which I happen to have written my thesis discussing!). A [Fermi surface](https://en.wikipedia.org/wiki/Fermi_surface) is a quantum mechanical structure that determines many of the electronic transport properties of a material (that is to say, the way electricity runs through it). For example, this surface will be different for metals, semi-conductors, and insulators. This discussion is beyond the scope of this demonstration, but free to shoot me an email if you're curious about learning about this.

In order to do this, we have taken angle-resolved magnetoresistance oscillation (AMRO) data, wherein we measure the longitudinal resistivity of a rectangular prism sample as we rotate it within a strong applied magnetic field. Resistivity is an intrinsic analogue to resistance, which you are likely more familiar with. This resistivity was measured in both positive and negative magnetic field values, and then symmetrized to minimize the influence of sample misalignment and any transverse resistivity that may be present.

We are interested in this because the symmetry of the AMRO (how many times the resistivity repeats per sample rotation), is determined by the Fermi surface. A significant change in the AMRO symmetry indicates a significant change in the Fermi surface structure. This is an unusual occurance and indicative of a significant change in the quantum physics of the material.

Investigations of such behaviour can help us discover new materials for use in technologies of the future.