# Poof of concept of predictive interlocks applied to the machine protection system for vacuum in particle accelerators and/or accelerator driven facilities: Inference pipeline

# Author: Dr. Zaera-Sanz, Manuel

# As main scenario, the use case of the ESS ERIC (European Spallation Source European Research Infrastructure Consortium) in Lund, is taken.

# The main contribution of this project is to increase the availability of proton beam, neutron beams and to provide cost savings (increment of time for experiments thanks to neutron beams availability). This is achieved by the prediction of interlocks situations caused by the vacuum system which involves stopping the machine and therefore stopping the proton beam production and neutrons generation for the experiments.

# The trained model already in production (training pipeline) is used to perform the inference obtaining a prediction (inference pipeline) 

# The user interface is done through a POST request which sends the new observation to the trained model, which performs the prediction accordingly. The new observation request uses JSON format
